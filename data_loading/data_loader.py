import torch
import numpy as np
import scipy.sparse as sp
import os
import pickle
from torch.utils.data import Dataset
from time import time


class DataLoader(Dataset):
    """
    Dataset type for pytorch \n
    Include graph information
    gowalla dataset
    """

    def __init__(self, opt: dict):
        self.opt = opt
        # train or test
        self.n_user = 0
        self.m_item = 0
        train_file = opt["data_path"] + '/train.txt'
        test_file = opt["data_path"] + '/test.txt'
        self.path = opt["data_path"]
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        validateItem, validateUser = [], []
        self.traindataSize = 0
        self.testDataSize = 0
        self.validateDataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip().split(' ')
                    items: list = list(set([int(i) for i in l[1:]]))
                    uid = int(l[0])

                    if len(items) != 0:
                        # take 1 / 3 of each user's original test set as validation set
                        validation_size = max(1, int(np.ceil((1 / 3) * len(items))))
                        valid_items = list(np.random.choice(items, size=validation_size, replace=False))
                        validateItem.extend(valid_items)
                        validateUser.extend([uid] * len(valid_items))
                        self.validateDataSize += len(valid_items)

                        test_items = set(items).difference(valid_items)
                        assert test_items.intersection(valid_items) == set()
                        test_items = list(test_items)
                        testUniqueUsers.append(uid)
                        testUser.extend([uid] * len(test_items))
                        testItem.extend(test_items)

                        # if no interacted items, don't change self.m_item
                        self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        self.validUser = np.array(validateUser)
        self.validItem = np.array(validateItem)

        print("Normalized adjacency graph loaded.")

        print(f"{self.train_data_size} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{self.validateDataSize} interactions for validation")
        print(
            f"{opt['dataset_name']} Sparsity : {(self.train_data_size + self.testDataSize) / self.n_users / self.m_items}")

        # (users,items), bipartite graph
        # NOTE: the binarized rating matrix for all *train* users and train items of shape (user#, item#)
        self.UserItemNet = sp.csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                         shape=(self.n_user, self.m_item))
        # assign normalized graph
        self.norm_adj_graph = None
        self.get_norm_graph()

        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()

        self.users_D[self.users_D == 0.] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        # all positive training items for each user (train_all_pos)
        self._allPos = self.get_user_pos_items(list(range(self.n_user)))
        # all positive testing items for each user
        self.__testDict = self.__build_test_or_validate(build_type="test")
        self.__validDict = self.__build_test_or_validate(build_type="valid")
        self.interact_mat = self.get_rating_matrix()

        print(f"{opt['dataset_name']} is ready to go")

    def get_rating_matrix(self, save_path=None):
        """
        Get rating matrix (bianrized rating matrix)

        * contains pos items from both train and test dataset
        """
        if save_path and os.path.exists(save_path):
            print("interaction matrix already exists")
            return

        # rating matrix contains only positively rated test items
        matrix_test_pos_only = sp.csr_matrix((np.ones(len(self.testUser)), (self.testUser, self.testItem)),
                                             shape=(self.n_user, self.m_item))
        matrix_valid_pos_only = sp.csr_matrix((np.ones(len(self.validUser)), (self.validUser, self.validItem)),
                                              shape=(self.n_user, self.m_item))
        # full rating matrix by performing bitwise or operation
        # on train and test positively rated items
        rating_matrix = self.UserItemNet + matrix_test_pos_only + matrix_valid_pos_only
        assert rating_matrix.min() == 0 and rating_matrix.max() == 1
        if save_path:
            sp.save_npz(save_path, rating_matrix)
            print(f"interaction matrix has been saved to {save_path}")
        return rating_matrix

    def save_test_pos_item_dict(self, save_path):
        """
        Save test positive items dict, in form {user: [positively rated items in test dataset]}
        """
        if os.path.exists(save_path):
            print("test_pos_item.pickle already exists")
            return
        with open(save_path, "wb") as fp:
            pickle.dump(self.test_all_pos, fp)
        print(f"test_pos_item.pickle dict has been saved to {save_path}")

    @property
    def n_users(self):
        """
        total #user in the dataset
        """
        return self.n_user

    @property
    def m_items(self):
        """
        total #item in the dataset
        """
        return self.m_item

    @property
    def train_data_size(self):
        """
        total #items in train.txt
        """
        return self.traindataSize

    @property
    def test_all_pos(self):
        """
        test_all_pos - uid: [all pos items in test dataset]
        """
        return self.__testDict

    @property
    def valid_all_pos(self):
        return self.__validDict

    @property
    def train_all_pos(self):
        """
        train_all_pos -- uid: [all pos items used in training]
        """
        return self._allPos

    def get_norm_graph(self):
        """
        Generate norm-adjacency matrix - shape (|U| + |I|) x (|U| + |I|),
        """
        print("loading adjacency matrix")
        if self.norm_adj_graph is None:
            try:
                norm_adj = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
            except:
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()

                end = time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

            # graph is a normalized adjacency matrix with all values lie within the range [0, 1]
            self.norm_adj_graph = norm_adj

    def __build_test_or_validate(self, build_type: str = "test"):
        """
        construct test dataset

        return:
            dict: {user: [pos rated items in test dataset]}
        """
        if build_type == "test":
            testItem = self.testItem
            testUser = self.testUser
        else:
            testItem = self.validItem
            testUser = self.validUser

        test_data = {}
        for i, item in enumerate(testItem):
            user = testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def get_user_item_feedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1,))

    def get_user_pos_items(self, users):
        """
        Get all *training* item IDs positively rated by users

        :return: each user's list of positive *training* item IDs
        e.g. [[1, 2, 3], [3, 5, 6]] means uid 0 has +ve items 1, 2, 3 and
            uid 1 has +ve items 3, 5, 6
        """
        pos_items = []
        for user in users:
            pos_items.append(self.UserItemNet[user].nonzero()[1])
        return pos_items

    # def getUserNegItems(self, users):
    #     negItems = []
    #     for user in users:
    #         negItems.append(self.allNeg[user])
    #     return negItems


def save_train_mapping(training_samples, save_path: str):
    """
    ** This saves all sampled (user, positive) mappings only, meaning
    not all Train positive items are included.

    Create train mapping and save it as a dict, each entry is of format (uid, pos_id): [neg id]

    :param training_samples: of the form {(uid, pos_id): [list of neg ids]}
    :param save_path: The path where train_sample.pickle is to be saved
    """
    if os.path.exists(save_path):
        return

    with open(save_path, "wb") as fp:
        pickle.dump(training_samples, fp)
    print(f"Training samples saved to {save_path}")


def save_all_positive_items(all_positive_lists: list, save_path: str):
    """
    Save list of all Train positive items.

    :param all_positive_lists: self._allPos
    :param save_path: the path where train_all_pos.pickle is to be saved
    """
    if os.path.exists(save_path):
        return

    with open(save_path, "wb") as fp:
        pickle.dump(all_positive_lists, fp)
    print(f"All training positive items are saved to {save_path}")


def generate_train_samples(dataloader: DataLoader, neg_size: int = 5) -> np.ndarray:
    """
    Generate training samples for each epoch.

    :param dataloader: the data loader used
    :param neg_size: the number of negative samples created for each (u, pos) pair.
    :return numpy array [(uid, +ve item, -ve item)] samples for training
    """
    user_num = dataloader.train_data_size
    users = np.random.randint(0, dataloader.n_users, user_num)
    train_all_pos = dataloader.train_all_pos

    # the array of (uid, +ve item, -ve item)
    samples = []
    # sampling strategy: for each (user, +ve), sample 1 -ve item for this pair
    for i, user in enumerate(users):
        user_train_pos = train_all_pos[user]
        if len(user_train_pos) == 0:
            continue
        pos_idx = np.random.randint(0, len(user_train_pos))
        pos_item = user_train_pos[pos_idx]

        negs = []
        while len(negs) != neg_size:
            while True:
                neg_item = np.random.randint(0, dataloader.m_items)
                if neg_item in user_train_pos or neg_item in negs:
                    continue
                else:
                    break
            negs.append(neg_item)
            samples.append([user, pos_item, neg_item])
    samples = np.array(samples)
    return samples


def minibatch(*tensors, **kwargs):
    """
    Create a minibatch for training
    """
    batch_size = kwargs.get('batch_size')

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):
    """
    Shuffle the train dataset instances
    """
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


def convert_sp_mat_to_sp_tensor(X):
    shape = torch.Size(X.shape)
    if not sp.isspmatrix_coo(X):
        X = X.tocoo(copy=False)
    if not X.dtype == np.float32:
        X = X.astype(np.float32, copy=False)
    row = torch.Tensor(X.row).long()
    col = torch.Tensor(X.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(X.data)
    return torch.sparse.FloatTensor(index, data, shape)


def convert_sp_tensor_to_sp_mat(X):
    if not X.is_sparse:
        X = X.to_sparse_coo()

    if X.is_cuda:
        X = X.cpu()
    shape = X.shape
    row, col = X.indices()
    vals = X.values()
    return sp.coo_matrix((vals, (row, col)), shape=shape).tocsr(copy=False)
