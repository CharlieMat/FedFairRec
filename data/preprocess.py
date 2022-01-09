
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm_multi_thread import TqdmMultiThreadFactory
import numpy as np
import time

def build_vocab(df, save_path, column_names):
    with open(save_path, 'w') as fout:
        fout.write("field_name\tvalue\tidx\n")
        for f in column_names:
            value_dict = {}
            for value in df[f]:
                for v in str(value).split(","):
                    if v not in value_dict:
                        value_dict[v] = len(value_dict) + 1 # index start from 1
            # save vocabulary
            for value, idx in value_dict.items():
                fout.write(f"{f}\t{value}\t{idx}\n")
    print(f"Vocab file saved to: {save_path}")

#######################################################
#                multicore filtering                  #
#######################################################
    

def repeat_n_core(df, user_n_core, item_n_core, user_counts, item_counts):
    '''
    Iterative n_core filter
    
    @input:
    - df: [UserID, ItemID, ...]
    - n_core: number of core
    - user_counts: {uid: frequency}
    - item_counts: {iid: frequency}
    '''
    print("N-core is set to [5,100]")
    user_n_core = min(max(user_n_core, 5),100) # 5 <= n_core <= 100
    item_n_core = min(max(item_n_core, 5),100) # 5 <= n_core <= 100
    print(f"Filtering ({user_n_core},{item_n_core})-core data")
    iteration = 0
    lastNRemove = len(df)  # the number of removed record
    proposedData = df.values
    originalSize = len(df)
    
    # each iteration, count number of records that need to delete
    while lastNRemove != 0:
        iteration += 1
        print("Iteration " + str(iteration))
        changeNum = 0
        newData = []
        for row in tqdm(proposedData):
            user, item = row[0], row[1]
            if user_counts[user] < user_n_core or item_counts[item] < item_n_core:
                user_counts[user] -= 1
                item_counts[item] -= 1
                changeNum += 1
            else:
                newData.append(row)
        proposedData = newData
        print("Number of removed record: " + str(changeNum))
        if changeNum > lastNRemove + 10000:
            print("Not converging, will use original data")
            break
        else:
            lastNRemove = changeNum
    print("Size change: " + str(originalSize) + " --> " + str(len(proposedData)))
    return pd.DataFrame(proposedData, columns=df.columns)
    
def run_multicore(df, n_core = 10, auto_core = False, filter_rate = 0.2):
    '''
    @input:
    - df: pd.DataFrame, col:[UserID,ItemID,...]
    - n_core: number of core
    - auto_core: automatically find n_core, set to True will ignore n_core
    - filter_rate: proportion of removal for user/item, require auto_core = True
    '''
    print(f"Filter {n_core if not auto_core else 'auto'}-core data.")
    uCounts = df["UserID"].value_counts().to_dict() # {user_id: count}
    iCounts = df["ItemID"].value_counts().to_dict() # {item_id: count}
            
    # automatically find n_core based on filter rate
    if auto_core:
        print("Automatically find n_core that filter " + str(100*filter_rate) + "% of user/item")
        
        nCoreCounts = dict() # {n_core: [#user, #item]}
        for v,c in iCounts.items():
            if c not in nCoreCounts:
                nCoreCounts[c] = [0,1]
            else:
                nCoreCounts[c][1] += 1
        for u,c in uCounts.items():
            if c not in nCoreCounts:
                nCoreCounts[c] = [1,0]
            else:
                nCoreCounts[c][0] += 1
                
        # find n_core for: filtered data < filter_rate * length(data)
        userToRemove = 0 # number of user records to remove
        itemToRemove = 0 # number of item records to remove
        for c,counts in sorted(nCoreCounts.items()):
            userToRemove += counts[0] * c # #user * #core
            itemToRemove += counts[1] * c # #item * #core
            if userToRemove > filter_rate * len(df) or itemToRemove > filter_rate * len(df):
                n_core = c
                print("Autocore = " + str(n_core))
                break
    else:
        print("n_core = " + str(n_core))
            
    return repeat_n_core(df, n_core, n_core, uCounts, iCounts)

def run_multicore_asymetric(df, n_core_user = 10, n_core_item = 10):
    '''
    @input:
    - df: pd.DataFrame, col:[UserID,ItemID,...]
    - n_core: number of core
    - auto_core: automatically find n_core, set to True will ignore n_core
    - filter_rate: proportion of removal for user/item, require auto_core = True
    '''
    uCounts = df["UserID"].value_counts().to_dict() # {user_id: count}
    iCounts = df["ItemID"].value_counts().to_dict() # {item_id: count}
    return repeat_n_core(df, n_core_user, n_core_item, uCounts, iCounts)


#################################################################################
#                           Train-val-test holdout                              #
#################################################################################


# Define a function for the thread
def holdout_thread(factory, df, thread_users, holdout_type, ratio, position, test_indices, val_indices):
    with factory.create(position, len(thread_users)) as progress:
#         print("Thread " + str(position) + " with " + str(len(thread_users)) + " users")
        for u in thread_users:
            userHistory = df[df["UserID"]==u].copy()
            if holdout_type == "leave_one_out":
                test_indices.iloc[userHistory.iloc[-1:].index] = True
                val_indices.iloc[userHistory.iloc[-2:-1].index] = True
            elif holdout_type == "warm":
                nTest = int(len(userHistory) * ratio[2])
                nVal = int(len(userHistory) * ratio[1])
                test_indices.iloc[userHistory.iloc[-nTest:].index] = True
                val_indices.iloc[userHistory.iloc[-nTest-nVal:-nTest].index] = True
            elif holdout_type == "cold":
                if np.random.random() < ratio[2]:
                    test_indices.iloc[userHistory.index] = True
                elif np.random.random() < (ratio[1] / (1. - ratio[2])):
                    val_indices.iloc[userHistory.index] = True
            progress.update(1)

def holdout_data(df, holdout_type = "warm", ratio = [0.8,0.1,0.1], n_worker = 2):
    '''
    Train-val-test hold out
    
    @input:
    - df: pd.DataFrame, [[UserId, ItemID, ..., timestamp]]
    - holdout_type: leave_one_out, warm, cold
    - 
    '''
#     print("Hold out {:.2f}-{:.2f}-{:.2f}".format(1-val_p-test_p, val_p, test_p))
    users = df["UserID"].unique()
    val_indices = df["UserID"]==-1
    test_indices = df["UserID"]==-1
    with ThreadPoolExecutor(max_workers=n_worker) as executor:
        factory = TqdmMultiThreadFactory()
        offset = len(users) // n_worker + 1
        for i in range(n_worker):
            executor.submit(holdout_thread, factory, df, users[i*offset: (i+1)*offset], 
                            holdout_type, ratio, i, test_indices, val_indices)
    
    testset = df[test_indices]
    valset = df[val_indices]
    trainset = df[~test_indices & ~val_indices]
    
    return trainset, valset, testset

def holdout_data_sequential(df, holdout_type = "warm", ratio = [0.8,0.1,0.1]):
    print("Build user history")
    user_hist = {}
    for pos,row in tqdm(enumerate(df.values)):
        u, *record = row
        if u not in user_hist:
            user_hist[u] = list()
        user_hist[u].append(pos)
    print("Holdout user histories")
    val_indices = df["UserID"]==-1
    test_indices = df["UserID"]==-1
    if holdout_type == "leave_one_out":
        '''
        Leave-one-out will split separate the last the 2ne last interaction of user history:
        test: last interaction of user history
        val: 2nd last interaction of user history
        val: remaining user history
        '''
        for u,H in tqdm(user_hist.items()):
            test_indices.iloc[H[-1:]] = True
            val_indices.iloc[H[-2:-1]] = True
    elif holdout_type == "warm":
        '''
        Warm holdout will split each user history by 8-1-1:
        train: first 80% of user history
        val: middle 10% of user history
        test: last 10% of user history
        '''
        for u,H in tqdm(user_hist.items()):
            nTest = max(int(len(H) * ratio[2]), 1)
            nVal = max(int(len(H) * ratio[1]), 1)
            test_indices.iloc[H[-nTest:]] = True
            val_indices.iloc[H[-nTest-nVal:-nTest]] = True
    elif holdout_type == "cold":
        '''
        Cold holdout will split users by 8-1-1:
        train: random 80% of user
        val: random 10% of user
        test: random 10% of user
        '''
        for u,H in tqdm(user_hist.items()):
            if np.random.random() < ratio[2]:
                test_indices.iloc[H] = True
            elif np.random.random() < (ratio[1] / (1. - ratio[2])):
                val_indices.iloc[H] = True
    testset = df[test_indices]
    valset = df[val_indices]
    trainset = df[~test_indices & ~val_indices]
    return trainset, valset, testset

def move_user_data(from_df, to_df, moving_user, user_hist, field_name):
    print("Moving user data")
    print(f"Before moving: Target DataFrame: {len(to_df)}, Source Data Frame: {len(from_df)}")
    moving_indices = from_df[field_name]==-1
    remain_indices = from_df[field_name]==-1
    # for all users that needs to move, include record indices in moving_indics
    for u in tqdm(moving_user):
        if u in user_hist:
            moving_indices[user_hist[u]] = True
            del user_hist[u]
    for u,H in tqdm(user_hist.items()):
        remain_indices[H] = True
    to_df = pd.concat([to_df, from_df[moving_indices]], axis = 0)
    from_df = from_df[remain_indices]  
    print(f"\n#user moved: {len(moving_user)}")
    print(f"After moving: Target DataFrame: {len(to_df)}, Source Data Frame: {len(from_df)}")
    return to_df, from_df

def recheck_exist(trainset, valset, testset, field_name = "ItemID"):
    '''
    This function ensures that all ids of "field_name" appear in trainset, and 
    there won't be unseen ids in valset or testset.
    '''
    print("Move unseen " + field_name + " from val to train")
    V = {v:1 for v in trainset[field_name].unique()} # all ids in train
    val_user_hist = {} # {uid: [row ids]}
    moving_user = {} # [uid], set of users to move from val to train
    pos = 0
    start_time = time.time()
    for u,i in zip(valset["UserID"], valset[field_name]):
        if u not in val_user_hist:
            val_user_hist[u] = list()
        val_user_hist[u].append(pos)
        if pos % 100000 == 0:
            time_spent = time.time() - start_time
            time_remain = time_spent * (len(valset) - pos) / (pos+1e-8)
            print("{}/{}, finish in {:.1f}s.   ".format(pos,len(valset),time_remain), end = '\r')
        pos += 1
        if i not in V: # move the user if it has unseen item and hasn't been counted
            moving_user[u] = 1
    moving_user = list(moving_user.keys())
    trainset, valset = move_user_data(from_df = valset, to_df = trainset, moving_user = moving_user, 
                                      user_hist = val_user_hist, field_name = field_name)
    print("Move unseen " + field_name + " from test to train, this may also move users in val to train")
    V = {v:1 for v in trainset[field_name].unique()} # updated ids in train
    for v in valset[field_name].unique():
        V[v] = 1
    test_user_hist = {} # {uid: [row_id]}
    moving_user = {} # [uid], set of users to move from test/val to train
    pos = 0
    start_time = time.time()
    for u,i in zip(testset["UserID"], testset[field_name]):
#         u,i,*record = testset.iloc[pos]
        if u not in test_user_hist:
            test_user_hist[u] = list()
        test_user_hist[u].append(pos)
        if pos % 100000 == 0:
            time_spent = time.time() - start_time
            time_remain = time_spent * (len(testset) - pos) / (pos+1e-8)
            print("{}/{}, finish in {:.1f}s.   ".format(pos,len(testset),time_remain), end = '\r')
        pos += 1
        if i not in V:
            moving_user[u] = 1
    moving_user = list(moving_user.keys())
    # also move val data along with test data
    print("Val --> Train")
    trainset, valset = move_user_data(from_df = valset, to_df = trainset, moving_user = moving_user, 
                                      user_hist = val_user_hist, field_name = field_name)
    print("Test --> Train")
    trainset, testset = move_user_data(from_df = testset, to_df = trainset, moving_user = moving_user, 
                                       user_hist = test_user_hist, field_name = field_name)
    return trainset, valset, testset