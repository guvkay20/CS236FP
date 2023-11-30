import rna_tools
import rna_tools.rna_tools_lib as rtl
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from model import RNADiffuser
import os
import copy
import random
import pickle
from torch.utils import tensorboard
#import tensorboard
import rmsd
import numpy as np
import pdb
from datetime import datetime

#random.seed(34)
#torch.manual_seed(34)

class MolGraph():
    def __init__(self,device):
        self.atoms = None # List; atom types
        self.adjList = None # Tensor of [numedges,2]
        self.coords = None # Tensor of  [numnodes,3]
        self.indices = None # Duple of 1D Tensors
        self.nbs = None
        self.nbe = None
        self.device = device

    def to(self,device):
        self.device = device
        return self

    def readFromPDB(self, pdbPath): # Needs Coords
        print(pdbPath)
        self.atoms = list()
        self.coords = list()
        adjList = list()
        
        try:
            seq, lines, rnas = loadRNA(pdbPath)
        except:
            return False
        if len(rnas.get_all_chain_ids()) > 1:
            return False
        self.seq = seq

        for acid in seq:
            if acid not in set(['A','C','G','U']):
                return False
        
        lines_by_res= [list() for _ in range(len(seq))] # NOTE INDEX OFFSET BY 1
        for line in lines:
            if line.split()[0]=="ATOM":
                try:
                    lines_by_res[rnas.get_res_num(line)-1].append(line)
                except:
                    return False
                self.atoms.append(rnas.get_atom_code(line))
                try:
                    self.coords.append(torch.tensor(rnas.get_atom_coords(line)).to(self.device))
                except:
                    return False
        self.coords = torch.stack(self.coords, dim=0)

        if len(self.atoms)>=10000:
            return False

        atoms_by_res = [dict() for _ in range(len(seq))] # Do go from atom numbs 1-indexed to 0-indexed 
        for i in range(len(seq)):
            for line in lines_by_res[i]:
                atoms_by_res[i][line.split()[2]] = rnas.get_atom_num(line)-1

            # TODO add assertions by residue type

        try:
            for i in range(len(seq)):
                if i > 0:
                    adjList.append(torch.tensor([atoms_by_res[i-1]["O3'"], atoms_by_res[i]["P"]]))

                adjList.append(torch.tensor([atoms_by_res[i]["P"], atoms_by_res[i]["OP1"]]))
                adjList.append(torch.tensor([atoms_by_res[i]["P"], atoms_by_res[i]["OP2"]]))
                adjList.append(torch.tensor([atoms_by_res[i]["P"], atoms_by_res[i]["O5'"]]))
                adjList.append(torch.tensor([atoms_by_res[i]["O5'"], atoms_by_res[i]["C5'"]]))
                adjList.append(torch.tensor([atoms_by_res[i]["C5'"], atoms_by_res[i]["C4'"]]))
                adjList.append(torch.tensor([atoms_by_res[i]["C4'"], atoms_by_res[i]["O4'"]]))
                adjList.append(torch.tensor([atoms_by_res[i]["C4'"], atoms_by_res[i]["C3'"]]))
                adjList.append(torch.tensor([atoms_by_res[i]["C3'"], atoms_by_res[i]["O3'"]]))
                adjList.append(torch.tensor([atoms_by_res[i]["C3'"], atoms_by_res[i]["C2'"]]))
                adjList.append(torch.tensor([atoms_by_res[i]["C2'"], atoms_by_res[i]["C1'"]]))
                adjList.append(torch.tensor([atoms_by_res[i]["C1'"], atoms_by_res[i]["O4'"]]))
    
                if (seq[i]=="A") or (seq[i]=="G"):
                    adjList.append(torch.tensor([atoms_by_res[i]["C1'"], atoms_by_res[i]["N9"]]))
                    adjList.append(torch.tensor([atoms_by_res[i]["N9"], atoms_by_res[i]["C8"]]))
                    adjList.append(torch.tensor([atoms_by_res[i]["C8"], atoms_by_res[i]["N7"]]))
                    adjList.append(torch.tensor([atoms_by_res[i]["N7"], atoms_by_res[i]["C5"]]))
                    adjList.append(torch.tensor([atoms_by_res[i]["C5"], atoms_by_res[i]["C4"]]))
                    adjList.append(torch.tensor([atoms_by_res[i]["N9"], atoms_by_res[i]["C4"]]))
    
                    adjList.append(torch.tensor([atoms_by_res[i]["C5"], atoms_by_res[i]["C6"]]))
                    adjList.append(torch.tensor([atoms_by_res[i]["C6"], atoms_by_res[i]["N1"]]))
                    adjList.append(torch.tensor([atoms_by_res[i]["N1"], atoms_by_res[i]["C2"]]))
                    adjList.append(torch.tensor([atoms_by_res[i]["C2"], atoms_by_res[i]["N3"]]))
                    adjList.append(torch.tensor([atoms_by_res[i]["N3"], atoms_by_res[i]["C4"]]))
    
                    if (seq[i]=="A"):
                        adjList.append(torch.tensor([atoms_by_res[i]["C6"], atoms_by_res[i]["N6"]]))
                    else:   # G by elimination
                        adjList.append(torch.tensor([atoms_by_res[i]["C6"], atoms_by_res[i]["O6"]])) 
                        adjList.append(torch.tensor([atoms_by_res[i]["C2"], atoms_by_res[i]["N2"]]))
    
                elif (seq[i]=="U") or (seq[i]=="C"):
                     adjList.append(torch.tensor([atoms_by_res[i]["C1'"], atoms_by_res[i]["N1"]]))
                     adjList.append(torch.tensor([atoms_by_res[i]["N1"], atoms_by_res[i]["C2"]]))
                     adjList.append(torch.tensor([atoms_by_res[i]["C2"], atoms_by_res[i]["O2"]]))
                     adjList.append(torch.tensor([atoms_by_res[i]["C2"], atoms_by_res[i]["N3"]]))
                     adjList.append(torch.tensor([atoms_by_res[i]["N3"], atoms_by_res[i]["C4"]]))
                     adjList.append(torch.tensor([atoms_by_res[i]["C4"], atoms_by_res[i]["C5"]]))
                     adjList.append(torch.tensor([atoms_by_res[i]["C5"], atoms_by_res[i]["C6"]]))
                     adjList.append(torch.tensor([atoms_by_res[i]["C6"], atoms_by_res[i]["N1"]]))
  
                     if (seq[i]=="C"):
                         adjList.append(torch.tensor([atoms_by_res[i]["C4"], atoms_by_res[i]["N4"]]))
                     else:   # U by elimination
                         adjList.append(torch.tensor([atoms_by_res[i]["C4"], atoms_by_res[i]["O4"]])) 
           

                else:
                    assert(False)
        except:
            return False
        
        self.adjList = torch.stack(adjList, dim=0).to(self.device)
        return True

    def generateIndices(self): # NOTE: Must have self.atoms
        if self.indices is None:
            self.indices = [list(),list()]
            for i in range(len(self.atoms)-1):
                self.indices[0].extend([i] * (len(self.atoms)-1-i))
                self.indices[1].extend(list(range(i+1,len(self.atoms))))

    def desparseEdges(self): # Note must have indices and self.adjList
        des = torch.zeros(len(self.indices[0]), dtype=torch.bool).to(self.device)
        lowerIndex = torch.min(self.adjList, dim=1).values
        higherIndex = torch.max(self.adjList, dim=1).values
        indexOffset = higherIndex - lowerIndex - 1
        indexBase =  len(self.atoms) * lowerIndex - (torch.pow(lowerIndex,2) + lowerIndex) / 2    #z -> zn - (z**2 + z) /2
        index = indexBase + indexOffset
        des[index.to(torch.int64)] = True
        return des
        
    def makeFromSeq(self, seq): # TODO, generate atoms and adjlist
        self.seq = seq
        pass

    def getNeighborsBySource(self): # (.,neigh)
        if self.nbs is None:
            neighsBySource = list()
            numNeighsBySource = list()
            for i in range(len(self.atoms)):
                neighsOfI = list(self.adjList[self.adjList[:,0]==i,1])
                neighsBySource.extend(neighsOfI)
                numNeighsBySource.append(len(neighsOfI))
            neighsBySource
            self.nbs = (neighsBySource, numNeighsBySource)
        return torch.tensor(self.nbs[0]).to(self.device), torch.tensor(self.nbs[1]).to(self.device)

    def getNeighborsByEnd(self):  # (neigh,.)
        if self.nbe is None:
            neighsByEnd = list()
            numNeighsByEnd = list()
            for i in range(len(self.atoms)):
                neighsOfI = list(self.adjList[self.adjList[:,1]==i,0])
                neighsByEnd.extend(neighsOfI)
                numNeighsByEnd.append(len(neighsOfI))
            self.nbe = (neighsByEnd, numNeighsByEnd)
        return torch.tensor(self.nbe[0]).to(self.device), torch.tensor(self.nbe[1]).to(self.device)

class MGDS(torch.utils.data.Dataset): # MolGraphDataSet
    def __init__(self, pdbPathsList, device):
       mgs = [MolGraph(device) for p in pdbPathsList] 
       res = [mg.readFromPDB(p) for mg,p in zip(mgs,pdbPathsList)]
       self.mgs = [mg for mg,r in zip(mgs,res) if r]
    
    def to(self, device):
        self.mgs = [mg.to(device) for mg in self.mgs]
        return self

    def __len__(self):
        return len(self.mgs)
        
    def __getitem__(self, idx):
        mg = self.mgs[idx]#copy.deepcopy(self.mgs[idx])
        #y = mg.coords
        #mg.coords = None
        return mg.atoms, mg.coords, mg.adjList, mg.seq#mg, y

def makeDSs(datafolder, device, cutoffs):  # device is relic, unused
    pdbs = [datafolder+"/"+f for f in os.listdir(datafolder) if (f[-4:]==".pdb")]
    random.shuffle(pdbs)

    c0 = int(float(len(pdbs)) * cutoffs[0])
    c1 = int(float(len(pdbs)) * cutoffs[1])

    train = MGDS(pdbs[:c0], 'cpu')
    valid = MGDS(pdbs[c0:c1], 'cpu')
    test = MGDS(pdbs[c1:], 'cpu')

    return train, valid, test

def recon(a, device):
    x,y,z,q = a
    y = y.to(device)
    z = z.to(device)
    X = MolGraph(device)
    X.atoms = x
    X.adjList = z
    X.coords = y
    X.seq = q
    X.generateIndices()
    return X
 

def getloss(model, batch, device):
    molgraphs = list()
    for (x,y,z,q) in batch:
        y = y.to(device)
        z = z.to(device)
        X = MolGraph(device)
        X.atoms = x
        X.adjList = z
        X.coords = y
        X.seq = q
        X.generateIndices()
        molgraphs.append(X)        

    #losses = list()
    #for molgraph in molgraphs:
    #    mg_loss =  model.computeDiffuserLoss(molgraph)
    #    losses.append(mg_loss)
    #loss = torch.mean(torch.stack(losses))
    loss = torch.mean(model.computeDiffuserLoss(molgraphs))

    return loss


def loadRNA(pdbPath):
    rnas = rna_tools.rna_tools_lib.RNAStructure(pdbPath)
    rnas.get_rnapuzzle_ready()
    seq = rnas.get_seq()
    lines = rnas.lines
    seq = seq.split("\n")[-1] 
    return seq, lines, rnas

def RMSD(coords1, coords2): # [num_atoms, 3] for each w same num_atoms
    c1 = coords1.numpy(force=True)
    c2 = coords2.numpy(force=True)
    
    c1 -= rmsd.centroid(c1)
    c2 -= rmsd.centroid(c2)

    U = rmsd.kabsch(c1,c2)
    c1 = np.dot(c1, U)

    return rmsd.rmsd(c1,c2)


def predictCoords(model, molGraphs, stepsAtNL=None):
    with torch.no_grad():
        coordsMatrices = model.generateCoords(molGraphs) if stepsAtNL is None else model.generateCoords(molGraphs, stepsAtNL)
    return coordsMatrices

def customLoader(valDS, factor, canSort):
    if canSort:
        order = list(range(len(valDS)))
        order.sort(key=lambda i : len(valDS[i][0]))
    else:
        order = list(range(len(valDS)))
        random.shuffle(order)
    batches = list()
    batch = list()
    ibatches = list()
    ibatch = list()
    maxleninbatch = 0
    maxTolerance = 11.5
    for i in order:
        item = valDS[i]
        if (len(item[0]) > 7000) or (len(item[0]) == 4953): # Memory, and special problem, respectively
            continue
        if (max(len(item[0])/1000,maxleninbatch)**2) *(1+len(batch)) * factor > maxTolerance:
            batches.append(batch)
            ibatches.append(ibatch)
            batch = list()
            ibatch = list()
            maxleninbatch = 0
        batch.append(item)
        ibatch.append(i)
        maxleninbatch = max(len(item[0])/1000,maxleninbatch)
    batches.append(batch)
    ibatches.append(ibatch)
    return batches

# NOTE use_ratio is deprecated; only changes recording
def validate(tp, model, validation_dataset, use_ratio, batch_size, device, sw=None, st=-1, useRMSD = False, log=True, pred_stepsAtNL=25): # If log, specify st
    #dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x:x)
    #pdb.set_trace()
    dataloader = customLoader(validation_dataset, factor=0.09, canSort=True)
    #goFor = 1 + int((use_ratio * len(validation_dataset) - 1.0) / float(batch_size))

    model.eval()
    with torch.no_grad():

        losses = []
        for batch_no, batch in enumerate(tqdm(dataloader, desc='Validating')):#, total=goFor)):
            #print(torch.cuda.memory_summary())
            #print([len(x) for x,y,z in batch])
            #if batch_no == goFor:
            #    break
            #break 
            loss = getloss(model, batch, device)

            losses.append(loss*len(batch))


        avg_loss = torch.tensor(sum(losses)/len(validation_dataset))

        

        #avg_rmsd = sum(rmsds)/len(rmsds)

        if log:
            if use_ratio < 1.0:
                sw.add_scalar('loss/intravalid',avg_loss,st)
            else:
                sw.add_scalar('loss/intervalid',avg_loss,st)
        print("Average Validation Loss", avg_loss.item())
       
        if useRMSD: #
            #rmsdDL = torch.utils.data.DataLoader(validation_dataset, batch_size=tp["predict_batchsize"], shuffle=True, collate_fn=lambda x:x)
            rmsdDL = customLoader(validation_dataset, factor=0.10, canSort=True)
            rmsds = []
            _rmsd = "N/A"
            for batch_no, batch in enumerate(tqdm(rmsdDL, desc=f"Validating RMSD, Last RMSD was {_rmsd}")):#, total=goFor)):
                #if batch_no == goFor:
                #    break
                #print(torch.cuda.memory_summary())
                #print([len(x) for x,y,z,q in batch])
                
                Xs = list()
                for (x,y,z,q) in batch:
                    y = y.to(device)
                    z = z.to(device)
                    X = MolGraph(device)
                    X.atoms = x
                    X.adjList = z
                    X.coords = y
                    X.seq = q
                    X.generateIndices()
                    Xs.append(X)

                coords_hats = predictCoords(model, Xs, pred_stepsAtNL)
                for X, coord_hat in zip(Xs, coords_hats):
                    X.predCoords = coord_hat
                    structureCode = str(random.randint(10000000,99999999))
                    torch.save(X, "predictions/predicted_structure_"+structureCode)
                    with open("predicthistory", "a") as f:
                        f.write(structureCode + " : " + str(X.seq) + "\n")
                    _rmsd = RMSD(coord_hat, X.coords)
                    rmsds.append(_rmsd)
                    print(_rmsd)
            print(rmsds)
            avg_rmsd = sum(rmsds)/len(rmsds) 
            print("Average Validation RMSD", avg_rmsd)
        
    model.train()
        
def test(model, test_dataset, batch_size, device):
    pass # TODO loss, RMSD; just like validate but use_ratio = 1.0

    

def train(hyper, tp, model, training_dataset, validation_dataset, # dataset is torch dataset
        device, batch_size, num_epochs, loaded_model_from_checkpoint=False):
    
    trainCode = str(datetime.now()).replace(" ","_")#random.randint(100000,999999)
    print("Training Code", trainCode)
    with open("trainhistory","a") as f:
        f.write(trainCode + " : " + str(hyper) + "\n")
    sw = tensorboard.SummaryWriter(log_dir="logs/"+str(trainCode)+"/")

    model.to(device)
    def initer(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.uniform_(m.bias)
        if isinstance(m, nn.Embedding):
            nn.init.xavier_normal_(m.weight)

    if not loaded_model_from_checkpoint:
        model.apply(initer)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyper["lr"])
    loss = torch.tensor([1.0])

    validate(tp, model, validation_dataset, 1.0, batch_size, device, sw, 0)
    for epoch in range(num_epochs):
        #dataloader = torch.utils.data.DataLoader(training_dataset, batch_size = batch_size, shuffle=True, collate_fn=lambda x: x )
        dataloader = customLoader(training_dataset, factor=0.22, canSort=False)

        for batch_no, batch in enumerate(tqdm(dataloader, desc=f'Training Iters, Epoch: {epoch}, CurLoss: {loss.to("cpu").item()}')):
            #print(torch.cuda.memory_summary())
            #print([len(x) for x,y,z,q in batch])
            optimizer.zero_grad()

            loss = getloss(model, batch, device)
            sw.add_scalar("loss/train",loss.to('cpu').item(),batch_no*batch_size+epoch*len(training_dataset))
            loss.backward()
            optimizer.step()

            # Limited benefit in intraepoch, IMO
            #if (batch_no*batch_size) % tp["intraepoch_val_every"] < batch_size:
            #    print("Intra Epoch Validation at Epoch", epoch, "Batch", batch_no)
            #    validate(tp, model, validation_dataset, tp["intraepoch_val_ratio"], tp["val_batchsize"], device, sw,batch_no*batch_size+epoch*len(training_dataset))

        
        print("End of Epoch",epoch,"\n")    
        validate(tp, model, validation_dataset, 1.0, batch_size, device, sw, epoch+1)

        torch.save(model.state_dict(),
            "checkpoints/train_endEpoch_"+str(trainCode)+"_epoch"+str(epoch)
        )
    torch.save(model.state_dict(),
        "checkpoints/train_final_"+str(trainCode)
    )


class Stat():
    def __init__(self, name, fxn):
        self.name = name
        self.apply = fxn


def DSstat(train, valid, test):
    stats = []

    stats.append(Stat("Len",lambda ds: len(ds)))

    def distribution(ds):
        numBucks = 100
        dt = [0 for _ in range(numBucks)]
        for item in ds:
            try:
                dt[int(len(item[0])/100)] += 1
            except:
                print("Overlong Seq Detected, Length:", len(item[0]))
                #raise Exception
        res = [(i*100,(i+1)*100-1,dt[i]) for i in range(numBucks)]
        res = [str(it) for it in res]
        return " ".join(res)
    stats.append(Stat("Distribution",distribution))

    for stat in stats:
        print("Train " + stat.name + ": " + str(stat.apply(train)))
        print("Validation " + stat.name + ": " + str(stat.apply(valid)))
        print("Test " + stat.name + ": " + str(stat.apply(test)))




def defaultTPs():
    return {
        "intraepoch_val_ratio":0.2,
        "intraepoch_val_every":80,
        "predict_steps_per_NL":10,
        "mode_val_ratio":1.0,
        "val_batchsize":4,
        "predict_batchsize":6,
    }

def defaultHypers():
    return {
        "mgin_conv_inlevel_layers":2,
        "mgin_conv_outlevel_layers":1,
        "mgin_layers":10,
        "noise_level_min":0.5,
        "noise_level_ratio":1.1,
        "noise_level_max":40.0,
        "embed_dims":3,
        "num_epochs":30,
        "mlp_in_layers":2,
        "mlp_mid_layers":2,
        "base_step_size": 2e-5,
        "gen_initial_std":40.0,
        "lr":1e-3,
        "train_batchsize":2
    }

def hp_search(trainDS, validDS, device):
    tp = defaultTPs()
    basehypers = defaultHypers()
    hyperranges = dict()
    for k,v in basehypers.items():
        hyperranges[k] = [v]
    
    # TODO custom mods on top
    # TODO only to variations
    # TODO fewer epochs

    # all combos
    #hypers = [dict()]
    #for k,r in hyperranges:
    #    newhypers = list()
    #    for hyper in hypers:
    #        for v in r:
    #            h = copy.deepcopy(hyper)
    #            h[k] = v
    #            newhypers.append(h)
    #    
    #    hypers = newhypers
    
    # each, separately
    hypers = []
    for k,r in hyperranges:
        for v in r:
            hyper = defaultHypers()
            hyper[k] = v
            hypers.append(hyper)

    for hyper in hypers:
        print("HP Search at: " + str(hyper))
        model = RNADiffuser(hyper, device)
        model.to(device)
        train(hyper, tp, model, trainDS, validDS, device, hyper["train_batchsize"], hyper["num_epochs"], False)


# Modes:
#   train: Trains Model, requires prepped DSs
#   validate: Runs validation (Loss + RMSD) on ValidDS, requires prepped DSs and trained model
#   test: Runs validation (Loss + RMSD) on TestDS, requires prepped DSs and trained model
#   make_dataset: Prepares 3-split DSs, requires PDBs in a folder, outputs joint DSs there
#   dataset_stat: Reports statistics for dataset; presently their length and atom-counts' distribution
#
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode")
    parser.add_argument("-d", "--dataset_folder", default="data/")
    parser.add_argument("-v", "--device", default=None)
    parser.add_argument("-m", "--model_path", default=None)
    args = parser.parse_args()

    if args.device is not None:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.mode=="make_dataset":
        trainDS, validDS, testDS = makeDSs(args.dataset_folder, args.device, (0.6,0.8))
        pickle.dump((trainDS, validDS, testDS), open(args.dataset_folder+"/DSs.pkl","wb"))
        quit()
    
    trainDS, validDS, testDS = pickle.load(open(args.dataset_folder+"/DSs.pkl","rb"))
    trainDS = trainDS.to(device)
    validDS = validDS.to(device)
    testDS = testDS.to(device)
    #trainDS.device = device
    #validDS.device = device
    #testDS.device = device

    if args.mode=="dataset_stat":
        DSstat(trainDS, validDS, testDS)
        quit()
    elif args.mode =="hp_search":
        hp_search(trainDS, validDS, device)
        quit()


    hyper = defaultHypers()
    tp = defaultTPs()

    model = RNADiffuser(hyper, device)
    loaded = False
    if args.model_path is not None:
        loaded = True
        model.load_state_dict(torch.load(args.model_path))
    model.to(device)
   
    if args.mode=="train":
        train(hyper, tp, model, trainDS, validDS, device, hyper["train_batchsize"], hyper["num_epochs"], loaded) #TODO if loaded, add random # and last epoch option to continue train smoothly
    elif args.mode=="validate":
        validate(tp, model, validDS, tp["mode_val_ratio"], tp["val_batchsize"], device, useRMSD = True, log=False, pred_stepsAtNL = tp["predict_steps_per_NL"]) # TODO 0.1 is a joke but required for rmsd realistic runtime
    elif args.mode=="predict":
        #print("RMSD",RMSD(predictCoords(model, recon(validDS[0], device), 1), validDS[0][1]))       # Currently set to debug prediction, TODO include sequence reconstruction
        print("RMSD",RMSD(torch.randn_like(validDS[0][1]).to(device), validDS[0][1]))
    elif args.mode=="test":
        print("Testing not implemented")
        raise Exception
    else:
        print("ERROR: Mode argument not found in options")
        assert(False)

