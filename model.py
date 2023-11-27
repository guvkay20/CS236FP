import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import math
import random
from tqdm import tqdm
#from main import molGraph

# SEQ -> NOISE -> STRUCTURE
# Look at ConfGF

class MGINConv(pyg.nn.MessagePassing):
    def __init__(self, channels):
        super().__init__('add')
        self.aggSum = nn.Sequential(
            nn.Linear(6,6),
            nn.LeakyReLU(),
            nn.Linear(6,6),
            nn.LeakyReLU(),
            nn.Linear(6,3),
            nn.LeakyReLU(),
            nn.Linear(3,3)
        ) 

    def forward(self, node_embeds, edge_embeds, edge_index):

        neigh_update = self.propagate(edge_index.T,x=node_embeds,edge_embeds=edge_embeds,id=torch.tensor(list(range(node_embeds.shape[0]))).reshape((-1,1)), num_nodes=node_embeds.shape[0])
        ret = self.aggSum(torch.cat((neigh_update,node_embeds),dim=1))
        return ret
        
        # new node embed = MLP(old node embed + sum across neighbors of ReLU(edge to neighbor embed + neighbor node embed))

    #def propagate(self, edge_index, size):
    
    def message (self, x_j,id_i,id_j, edge_embeds,num_nodes):
        
        a = torch.min(torch.stack((id_i,id_j),dim=1),dim=1).values.to(torch.float64)
        b = torch.max(torch.stack((id_i,id_j),dim=1),dim=1).values.to(torch.float64)
        indices = ((b-1)+a * (num_nodes - (a * 0.5) - 1.5)).to(torch.int64).reshape((-1))
        existing_edge_embeds = torch.index_select(edge_embeds,0,indices)

        return F.relu(x_j + existing_edge_embeds)
    #def aggregate(inputs, index)


class ModifiedGIN(nn.Module):
    def __init__(self, channels, num_layers):
        super().__init__()
        self.conv = nn.ModuleList([MGINConv(channels) for _ in range(num_layers)])  

    def forward(self, node_embeds, edge_embeds, edge_index):
        for conv in self.conv:
            node_embeds = conv(node_embeds, edge_embeds, edge_index) 
        return node_embeds

class RNADiffuser(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.embedMap = dict()
        self.nodeEmbedder = nn.Embedding(100, 3)
        self.edgeEmbedder = nn.Linear(4, 3)
        self.edgeNetwork = ModifiedGIN(3,10)
        self.finalEmbedder = nn.Sequential(
                nn.Linear(9,9),
                nn.LeakyReLU(),
                nn.Linear(9,9),
                nn.LeakyReLU(),
                nn.Linear(9,3),
                nn.LeakyReLU(),
                nn.Linear(3,3),
                nn.LeakyReLU(),
                nn.Linear(3,3),
                nn.LeakyReLU(),
                nn.Linear(3,1)
        )
        
        self.noiseLevels = []
        nl = 40.0
        ratio = 1.1
        while nl > 0.5:
            self.noiseLevels.append(nl)
            nl /= ratio

        self.baseStepSize = 2e-5

    def forward(self, molGraph, d_sim):
        # molGraph format:
        #    molGraph.atoms -> List of Atom Types, indices are indices
        #    molGraph.adjList -> List of connections
        #    molGraph.indices -> Duple of indices
        #                 [0] -> 0 n-1 times, 1 n-2 times, 2 n-3 times ... n-2 1 time
        #                 [1] -> 1,2,...n-1,2,3,...n-1,....n-1,n-2,n-1,n-1
        #
        # d_sim should be 1D vector of raster-order distances of strict upper triangle of atomsXatoms matrix

        # atoms tokenizer; allocate up to 100 molecules dynamically
        def conv(atom):
            if atom not in self.embedMap:
                self.embedMap[atom] = len(self.embedMap)
            return self.embedMap[atom]
        tokAtoms = torch.tensor([conv(at) for at in molGraph.atoms]).to(molGraph.adjList.device)

        # embed each node (by type? paper does by charge but IDT it is necessary for tihis); use a MLP???;
        node_embeds = self.nodeEmbedder(tokAtoms)

        # embed each edge in adjlist using predicted dist; use a MLP???
        edge_exists = 1 * molGraph.desparseEdges()
        edge_embeds = self.edgeEmbedder(torch.stack([edge_exists, d_sim, torch.sqrt(torch.abs(d_sim)), torch.pow(d_sim, 2)], dim=1))

        # standard message passing network with sum as agg basis, and embeds of edge and neighbor added for each message, self-con included; message computed directly, aggregate subject to MLP
        final_node_embeds = self.edgeNetwork(node_embeds, edge_embeds, molGraph.adjList)

        # after network, concat source || neighbor || edge embeds, use final to the nodes; subject to MLP, get edge scores in 3D
        LHS = torch.index_select(final_node_embeds, 0, torch.tensor(molGraph.indices[0]))# 0 n-1 times, 1 n-2 times, 2 n-3 times ... n-2 1 time
        RHS = torch.index_select(final_node_embeds, 0, torch.tensor(molGraph.indices[1]))#1,2,...n-1,2,3,...n-1,....n-1,n-2,n-1,n-1
        
        concat_embeds = torch.cat((LHS,RHS,edge_embeds),dim=1)
        scores = self.finalEmbedder(concat_embeds)
       
        return scores.reshape((-1))


    # NOTE: Can compute one graph at a time, step batching should be done in main
    def computeDiffuserLoss(self, molGraph):#, noiseLevels):
        # molGraph format:
        #    molGraph.atoms -> List of Atom Types, indices are indices
        #    molGraph.adjList -> List of connections
        #    molGraph.coords -> Tensor [atom_num X 3]
        #
        # noiseLevels -> list of floats [[ No Longer]]
    
        #losses = list()

        #for noise_level in noiseLevels:
        #with random.choice(self.noiseLevels) as noise_level:  # TODO noise level as input to score fxn; maybe not
        noise_level = random.choice(self.noiseLevels)
    
        with torch.no_grad():
            d = self.computeDVector(molGraph.coords, molGraph.indices)
            
            d_sim = d + torch.normal(torch.zeros(d.shape), torch.ones(d.shape) * noise_level)
                

        scores = self.forward(molGraph, d_sim)
        
        loss_at_nl = noise_level**2 * torch.pow(torch.linalg.vector_norm((scores / noise_level) + ((d_sim - d) / noise_level**2)), 2)
            #losses.append(loss_at_nl)
        return loss_at_nl
        #return torch.mean(torch.stack(losses))

    # NOTE: Don't forget to turn on eval mode here for no grad
    def generateCoords(self, molGraph, stepsAtNL=25):
        # molGraph format:
        #    molGraph.atoms -> List of Atom Types, indices are indices
        #    molGraph.adjList -> List of connections

        molGraph.generateIndices()
        sourceds, sourceds_lens = molGraph.getNeighborsBySource() # These don't get inverted
        enders, enders_lens = molGraph.getNeighborsByEnd() # These get inverted
        psums_sl = torch.cat([torch.tensor([0]),torch.cumsum(sourceds_lens,0) ], dim=0)
        psums_el = torch.cat([torch.tensor([0]),torch.cumsum(enders_lens,0) ], dim=0)

        R = torch.normal(torch.zeros((len(molGraph.atoms),3)), 10)
        iters = 0
        with tqdm(total=len(self.noiseLevels)*stepsAtNL, desc=f"Generating Molecule Structure") as pbar:
            #print("Noise Levels", self.noiseLevels)
            for noise_level in self.noiseLevels:
                #print("Current Noise Level:", noise_level)
                step_size = self.baseStepSize * (noise_level**2) / (self.noiseLevels[-1]**2)
                for t in range(stepsAtNL):
                    pbar.update(1)
                    d = self.computeDVector(R, molGraph.indices) 
                    
                    scores = self.forward(molGraph, d)
                    normscores = scores / d
                    
                    deltaRs = self.pairwiseDisplacements(R, molGraph.indices) 
                    fixscores = (deltaRs.T * normscores).T
                    # fixscores in raster order
                    # NOTE fixscores is (i-j), where i is source node and j are neighbors
                    
                    # compute s by gathering for each index s the fixscores for those connected to s and summing them and negating them where needed
                
                    sourced_scores = torch.index_select(fixscores, 0, sourceds)
                    ender_scores = torch.index_select(fixscores, 0, enders) * -1
                    
                    ss = list()
                    for i in range(len(sourceds_lens)):
                        ss.append(
                                torch.sum(sourced_scores[psums_sl[i] : psums_sl[i+1], :], dim=0) #torch.sum(sourced_scores[sum(sourceds_lens[:i]) : sum(sourceds_lens[:i+1]), :], dim=0) 
                                +
                                torch.sum(ender_scores[psums_el[i] : psums_el[i+1], :], dim=0)#torch.sum(ender_scores[sum(enders_lens[:i]) : sum(enders_lens[:i+1]), :], dim=0) 
                        )
                    s = torch.stack(ss, dim=0) /noise_level
            

                    # Compute s theta
                    z = torch.normal(torch.zeros((1)),torch.ones((1)))
                    R = R + step_size * s + z * math.sqrt(2 * step_size)
        return R

    def pairwiseDisplacements(self, coords, indices): # raster order elemwise deltas
        # 0 to 1...n-1
        # 1 to 2...n-1
        # ...
        # n-2 to n-1
        # all in 1D
        
        LHS = torch.index_select(coords, 0, torch.tensor(indices[0]))# 0 n-1 times, 1 n-2 times, 2 n-3 times ... n-2 1 time
        RHS = torch.index_select(coords, 0, torch.tensor(indices[1]))#1,2,...n-1,2,3,...n-1,....n-1,n-2,n-1,n-1

        return LHS - RHS

    def computeDVector(self, coords, indices): # use below fxn; return raster orders dists in 1D vector
        pds = self.pairwiseDisplacements(coords, indices)
        return torch.sqrt(torch.sum(torch.pow(pds, 2),dim=1))


