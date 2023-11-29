import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import math
import random
from tqdm import tqdm
import pdb
#from main import molGraph

# SEQ -> NOISE -> STRUCTURE
# Look at ConfGF

class MGINConv(pyg.nn.MessagePassing):
    def __init__(self, channels, hyper, device):
        super().__init__('add')
        self.device = device

        self.aggSum = nn.Sequential(
                *([nn.Linear(2*channels,2*channels),nn.LeakyReLU()] *hyper["mgin_conv_inlevel_layers"]
                +
                [nn.Linear(2*channels,channels)]
                +
                [nn.LeakyReLU(),nn.Linear(channels,channels)] *hyper["mgin_conv_outlevel_layers"])
        ) 

    def forward(self, node_embeds, edge_embeds, edge_index, cutoffs, graph_lens):
        #pdb.set_trace()
        neigh_update = self.propagate(edge_index.T,x=node_embeds,edge_embeds=edge_embeds,id=torch.tensor(list(range(node_embeds.shape[0]))).to(self.device).reshape((-1,1)), num_nodes=graph_lens,cutoffs=cutoffs)
        ret = self.aggSum(torch.cat((neigh_update,node_embeds),dim=1))
        return ret
        
        # new node embed = MLP(old node embed + sum across neighbors of ReLU(edge to neighbor embed + neighbor node embed))

    #def propagate(self, edge_index, size):
    
    def message (self, x_j,id_i,id_j, edge_embeds,num_nodes,cutoffs):
        #pdb.set_trace() 
        cutoffs = torch.cat([cutoffs, torch.sum(num_nodes).unsqueeze(0)], dim=0)
        a = torch.min(torch.stack((id_i,id_j),dim=1),dim=1).values.to(torch.float64)
        b = torch.max(torch.stack((id_i,id_j),dim=1),dim=1).values.to(torch.float64)
        
        index_in_graph_index = torch.sum(a >= cutoffs[1:], dim=1)
        intrasize = num_nodes * (num_nodes - 1) / 2
        incr_by = torch.cumsum(torch.cat([torch.tensor([0]).to(self.device),intrasize],dim=0)[:-1], dim=0)
        a_in = a.squeeze(1) - cutoffs[index_in_graph_index]
    
        indices = a_in * (num_nodes[index_in_graph_index] - (a_in * 0.5) - 1.5) + b.squeeze(1) - 1 - cutoffs[index_in_graph_index] + incr_by[index_in_graph_index]
        indices = indices.to(torch.int64)

        existing_edge_embeds = torch.index_select(edge_embeds,0,indices)

        return F.relu(x_j + existing_edge_embeds)
    #def aggregate(inputs, index)


class ModifiedGIN(nn.Module):
    def __init__(self, channels, hyper, device):
        super().__init__()
        self.device = device
        self.conv = nn.ModuleList([MGINConv(channels,hyper,device) for _ in range(hyper["mgin_layers"])])  

    def forward(self, node_embeds, edge_embeds, molGraphs):
        #pdb.set_trace()
        node_embeds = torch.cat(nn.utils.rnn.unpad_sequence(node_embeds, torch.tensor([len(mg.atoms) for mg in molGraphs]).to(self.device),batch_first=True), dim=0)
        edge_embeds = torch.cat(nn.utils.rnn.unpad_sequence(edge_embeds, torch.tensor([len(mg.indices[0]) for mg in molGraphs]).to(self.device),batch_first=True), dim=0)
        incr_by = torch.cumsum(torch.tensor([0] + [len(mg.atoms) for mg in molGraphs]).to(self.device)[:-1], dim=0)
        edge_index = torch.cat([mg.adjList+incr for incr,mg in zip(incr_by,molGraphs)], dim=0)

        for conv in self.conv:
            node_embeds = conv(node_embeds, edge_embeds, edge_index, incr_by, torch.tensor([len(mg.atoms) for mg in molGraphs]).to(self.device)) 
        return torch.nn.utils.rnn.pad_sequence(torch.split(node_embeds, [len(mg.atoms) for mg in molGraphs]), batch_first=True)

class RNADiffuser(nn.Module):
    def __init__(self, hyper, device):
        super().__init__()
        self.device = device

        self.embedMap = dict()
        self.nodeEmbedder = nn.Embedding(100, hyper["embed_dims"], padding_idx=0)
        self.edgeEmbedder = nn.Linear(4, hyper["embed_dims"])
        self.edgeNetwork = ModifiedGIN(hyper["embed_dims"],hyper,device)
        self.finalEmbedder = nn.Sequential(*(
                [nn.Linear(3*hyper["embed_dims"],3*hyper["embed_dims"]),nn.LeakyReLU()] * hyper["mlp_in_layers"] +
                [nn.Linear(3*hyper["embed_dims"],hyper["embed_dims"])] +
                [nn.LeakyReLU(),nn.Linear(hyper["embed_dims"],hyper["embed_dims"])] * hyper["mlp_mid_layers"] +
                [nn.LeakyReLU(),nn.Linear(hyper["embed_dims"],1)]
        ))
        
        self.noiseLevels = []
        nl = hyper["noise_level_max"]
        ratio = hyper["noise_level_ratio"]
        while nl > hyper["noise_level_min"]:
            self.noiseLevels.append(nl)
            nl /= ratio

        self.baseStepSize = hyper["base_step_size"]
        self.initialDeviation = hyper["gen_initial_std"]

        self.hyper = hyper

    def forward(self, molGraphs, d_sims):
        # molGraph format:
        #    molGraph.atoms -> List of Atom Types, indices are indices
        #    molGraph.adjList -> List of connections
        #    molGraph.indices -> Duple of indices
        #                 [0] -> 0 n-1 times, 1 n-2 times, 2 n-3 times ... n-2 1 time
        #                 [1] -> 1,2,...n-1,2,3,...n-1,....n-1,n-2,n-1,n-1
        #
        # d_sim should be 1D vector of raster-order distances of strict upper triangle of atomsXatoms matrix

        # atoms tokenizer; allocate up to 100 molecules dynamically
        #pdb.set_trace()
        def conv(atom):
            if atom not in self.embedMap:
                self.embedMap[atom] = len(self.embedMap)+1
            return self.embedMap[atom]
        tokAtomss = list()
        for molGraph in molGraphs:
            tokAtoms = torch.tensor([conv(at) for at in molGraph.atoms]).to(molGraph.adjList.device)
            tokAtomss.append(tokAtoms)
        tokAtoms = torch.nn.utils.rnn.pad_sequence(tokAtomss, batch_first=True)

        # embed each node (by type? paper does by charge but IDT it is necessary for tihis); use a MLP???;
        node_embeds = self.nodeEmbedder(tokAtoms) #B, max(L), H (embedding dim)

        # embed each edge in adjlist using predicted dist; use a MLP???
        edge_exists = torch.nn.utils.rnn.pad_sequence([1 * molGraph.desparseEdges() for molGraph in molGraphs], batch_first=True)
        edge_embeds = self.edgeEmbedder(torch.stack([edge_exists, d_sims, torch.sqrt(torch.abs(d_sims)), torch.pow(d_sims, 2)], dim=2)) # B, max(:d), H

        # standard message passing network with sum as agg basis, and embeds of edge and neighbor added for each message, self-con included; message computed directly, aggregate subject to MLP
        final_node_embeds = self.edgeNetwork(node_embeds, edge_embeds, molGraphs)

        # after network, concat source || neighbor || edge embeds, use final to the nodes; subject to MLP, get edge scores in 3D
        LHS = torch.nn.utils.rnn.pad_sequence([torch.index_select(final_node_embeds[i], 0, torch.tensor(molGraph.indices[0]).to(self.device)) for i,molGraph in enumerate(molGraphs)], batch_first=True)# 0 n-1 times, 1 n-2 times, 2 n-3 times ... n-2 1 time
        RHS = torch.nn.utils.rnn.pad_sequence([torch.index_select(final_node_embeds[i], 0, torch.tensor(molGraph.indices[1]).to(self.device)) for i,molGraph in enumerate(molGraphs)], batch_first=True)#1,2,...n-1,2,3,...n-1,....n-1,n-2,n-1,n-1
        
        concat_embeds = torch.cat((LHS,RHS,edge_embeds),dim=2)
        scores = self.finalEmbedder(concat_embeds)
       
        return scores.squeeze(2)


    # NOTE: Can compute one graph at a time, step batching should be done in main
    def computeDiffuserLoss(self, molGraphs):#, noiseLevels):
        # molGraph format:
        #    molGraph.atoms -> List of Atom Types, indices are indices
        #    molGraph.adjList -> List of connections
        #    molGraph.coords -> Tensor [atom_num X 3]
        #
        # noiseLevels -> list of floats [[ No Longer]]
    
        #losses = list()

        #for noise_level in noiseLevels:
        #with random.choice(self.noiseLevels) as noise_level:  # TODO noise level as input to score fxn; maybe not
        #pdb.set_trace()
        noise_levels = torch.tensor(random.choices(self.noiseLevels, k=len(molGraphs))).to(self.device)
    
        with torch.no_grad():
            ds,vals = self.computeDVectors(molGraphs)#molGraph.coords, molGraph.indices)
            
            d_sims = ds + torch.normal(torch.zeros(ds.shape).to(self.device), torch.ones(ds.shape).to(self.device) * noise_levels.unsqueeze(1))
                

        scores = self.forward(molGraphs, d_sims)
        
        losses_at_nl = torch.pow(noise_levels,2) * torch.pow(torch.linalg.vector_norm((scores*vals / noise_levels.unsqueeze(1)) + ((d_sims - ds)*vals / torch.pow(noise_levels.unsqueeze(1),2)), dim=1), 2)
            #losses.append(loss_at_nl)
        return losses_at_nl
        #return torch.mean(torch.stack(losses))

    # NOTE: Don't forget to turn on eval mode here for no grad
    def generateCoords(self, molGraphs, stepsAtNL=25):
        # molGraph format:
        #    molGraph.atoms -> List of Atom Types, indices are indices
        #    molGraph.adjList -> List of connections

        [molGraph.generateIndices() for molGraph in molGraphs]
###
        edgeStuffs = list()
        for i in range(len(molGraphs)):
            molGraph = molGraphs[i]
            sourceds, sourceds_lens = molGraph.getNeighborsBySource() # These don't get inverted
            enders, enders_lens = molGraph.getNeighborsByEnd() # These get inverted
            psums_sl = torch.cat([torch.tensor([0]).to(self.device),torch.cumsum(sourceds_lens,0) ], dim=0)
            psums_el = torch.cat([torch.tensor([0]).to(self.device),torch.cumsum(enders_lens,0) ], dim=0)
            edgeStuffs.append((sourceds,sourceds_lens,enders,enders_lens,psums_sl,psums_el))
###
        R = torch.normal(torch.zeros((len(molGraphs),max([len(molGraph.atoms) for molGraph in molGraphs]),3)).to(self.device), self.initialDeviation) # 10 is estimated radius
        iters = 0
        with tqdm(total=len(self.noiseLevels)*stepsAtNL, desc=f"Generating Molecule Structure") as pbar:
            #print("Noise Levels", self.noiseLevels)
            for noise_level in self.noiseLevels:
                #print("Current Noise Level:", noise_level)
                step_size = self.baseStepSize * (noise_level**2) / (self.noiseLevels[-1]**2)                
                for t in range(stepsAtNL):
                    pbar.update(1)
                    ds = self.computeDVectors_(R, molGraphs) 
                    scores = self.forward(molGraphs, ds)
                    normscores = scores / ds
                    
                    deltaRs = torch.nn.utils.rnn.pad_sequence([self.pairwiseDisplacements(R[i], molGraphs[i].indices) for i in range(len(molGraphs))], batch_first=True)
                    fixscores = deltaRs * normscores.unsqueeze(2)
                    # fixscores in raster order
                    # NOTE fixscores is (i-j), where i is source node and j are neighbors
                    
                    # compute s by gathering for each index s the fixscores for those connected to s and summing them and negating them where needed

##                
                    s = list()
                    for i in range(len(molGraphs)):
                        sourceds,sourceds_lens,enders,enders_lens,psums_sl,psums_el = edgeStuffs[i]
                        sourced_scores = torch.index_select(fixscores[i], 0, sourceds)
                        ender_scores = torch.index_select(fixscores[i], 0, enders) * -1
                    
                        ss = list()
                        for i in range(len(sourceds_lens)):
                            ss.append(
                                torch.sum(sourced_scores[psums_sl[i] : psums_sl[i+1], :], dim=0) #torch.sum(sourced_scores[sum(sourceds_lens[:i]) : sum(sourceds_lens[:i+1]), :], dim=0) 
                                +
                                torch.sum(ender_scores[psums_el[i] : psums_el[i+1], :], dim=0)#torch.sum(ender_scores[sum(enders_lens[:i]) : sum(enders_lens[:i+1]), :], dim=0) 
                        )
                        sk = torch.stack(ss, dim=0) /noise_level
                        s.append(sk)
                    s = torch.nn.utils.rnn.pad_sequence(s,batch_first=True)
###
                    # Compute s theta
                    z = torch.normal(torch.zeros(R.shape),torch.ones(R.shape)).to(self.device)
                    R = R + step_size * s + z * math.sqrt(2 * step_size)
        #pdb.set_trace()
        R = torch.nn.utils.rnn.unpad_sequence(R, torch.tensor([len(mg.atoms)  for mg in molGraphs]).to(self.device), batch_first=True)
        return R

    def pairwiseDisplacements(self, coords, indices): # raster order elemwise deltas
        # 0 to 1...n-1
        # 1 to 2...n-1
        # ...
        # n-2 to n-1
        # all in 1D
        
        LHS = torch.index_select(coords, 0, torch.tensor(indices[0]).to(self.device))# 0 n-1 times, 1 n-2 times, 2 n-3 times ... n-2 1 time
        RHS = torch.index_select(coords, 0, torch.tensor(indices[1]).to(self.device))#1,2,...n-1,2,3,...n-1,....n-1,n-2,n-1,n-1

        return LHS - RHS

    def computeDVector(self, coords, indices): # use below fxn; return raster orders dists in 1D vector
        pds = self.pairwiseDisplacements(coords, indices)
        return torch.linalg.vector_norm(pds, dim=1)#torch.sqrt(torch.sum(torch.pow(pds, 2),dim=1))

    def computeDVectors(self, molGraphs):
        pdss = [self.pairwiseDisplacements(molGraph.coords, molGraph.indices) for molGraph in molGraphs]
        padpdss = nn.utils.rnn.pad_sequence(pdss, batch_first = True)
        valpdss = [1+(pds*0) for pds in pdss]
        vals = nn.utils.rnn.pad_sequence(valpdss, batch_first=True).to(torch.bool) * 1 
        return torch.linalg.vector_norm(padpdss, dim=2), vals[:,:,0]

    def computeDVectors_(self, coordss, molGraphs):
        pdss = [self.pairwiseDisplacements(coords, mg.indices) for coords, mg in zip(coordss, molGraphs)]
        padpdss = nn.utils.rnn.pad_sequence(pdss, batch_first = True)
        #valpdss = [1+(pds*0) for pds in pdss]
        #vals = nn.utils.rnn.pad_sequence(valpdss, batch_first=True).to(torch.bool) * 1 
        return torch.linalg.vector_norm(padpdss, dim=2)#, vals


