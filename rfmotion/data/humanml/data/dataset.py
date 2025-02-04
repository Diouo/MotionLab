import codecs as cs
import os
import random
import math
from os.path import join as pjoin

import numpy as np
import spacy
import torch
from rich.progress import track
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from ..utils.get_opt import get_opt
from ..utils.word_vectorizer import WordVectorizer


# import spacy
def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


"""For use of training text-2-motion generative model"""


class Text2MotionDataset(data.Dataset):

    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        min_motion_len = 40 if self.opt.dataset_name == "t2m" else 24

        joints_num = opt.joints_num

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + ".npy"))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20):int(to_tag *
                                                                      20)]
                                if (len(n_motion)) < min_motion_len or (
                                        len(n_motion) >= 200):
                                    continue
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
                                    "_" + name)
                                while new_name in data_dict:
                                    new_name = (random.choice(
                                        "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
                                                name)
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                      to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                # Some motion may not exist in KIT dataset
                pass

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4:4 + (joints_num - 1) * 3] = std[4:4 +
                                                  (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3:4 +
                (joints_num - 1) * 9] = (std[4 + (joints_num - 1) * 3:4 +
                                             (joints_num - 1) * 9] / 1.0)
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9:4 + (joints_num - 1) * 9 +
                joints_num * 3] = (std[4 + (joints_num - 1) * 9:4 +
                                       (joints_num - 1) * 9 + joints_num * 3] /
                                   1.0)
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = (
                std[4 +
                    (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias)

            assert 4 + (joints_num -
                        1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, "mean.npy"), mean)
            np.save(pjoin(opt.meta_dir, "std.npy"), std)

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.opt.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data[
            "text"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if self.opt.is_train:
            if m_length != self.max_length:
                # print("Motion original length:%d_%d"%(m_length, len(motion)))
                if self.opt.unit_length < 10:
                    coin2 = np.random.choice(["single", "single", "double"])
                else:
                    coin2 = "single"
                if len_gap == 0 or (len_gap == 1 and coin2 == "double"):
                    m_length = self.max_length
                    idx = random.randint(0, m_length - self.max_length)
                    motion = motion[idx:idx + self.max_length]
                else:
                    if coin2 == "single":
                        n_m_length = self.max_length + self.opt.unit_length * len_gap
                    else:
                        n_m_length = self.max_length + self.opt.unit_length * (
                            len_gap - 1)
                    idx = random.randint(0, m_length - n_m_length)
                    motion = motion[idx:idx + self.max_length]
                    m_length = n_m_length
                # print(len_gap, idx, coin2)
        else:
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(["single", "single", "double"])
            else:
                coin2 = "single"

            if coin2 == "double":
                m_length = (m_length // self.opt.unit_length -
                            1) * self.opt.unit_length
            elif coin2 == "single":
                m_length = (m_length //
                            self.opt.unit_length) * self.opt.unit_length
            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx:idx + m_length]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length


"""For use of training text motion matching model, and evaluations"""


class Text2MotionDatasetV2(data.Dataset):

    def __init__(
        self,
        mean,
        std,
        split_file,
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        motion_dir,
        text_dir,
        tiny=False,
        debug=False,
        progress_bar=True,
        **kwargs,
    ):
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        # min_motion_len = 40 if dataset_name =='t2m' else 24
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        self.id_list = id_list

        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10

        if progress_bar:
            enumerator = enumerate(
                track(
                    id_list,
                    f"Loading HumanML3D {split_file.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator = enumerate(id_list)
        count = 0
        bad_count = 0
        new_name_list = []
        length_list = []
        for i, name in enumerator:
            if count > maxdata:
                break
            try:
                motion = np.load(pjoin(motion_dir, name + ".npy"))
                if (len(motion)) < self.min_motion_length or (len(motion) >=200):
                    bad_count += 1
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20):int(to_tag * 20)]
                                if (len(n_motion)) < self.min_motion_length or ((len(n_motion) >= 200)):
                                    continue
                                new_name = (random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +"_" + name)
                                while new_name in data_dict:
                                    new_name = (random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" +name)
                                data_dict[new_name] = {"motion": n_motion,"length": len(n_motion),"text": [text_dict],}
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,to_tag, name)

                if flag:
                    data_dict[name] = {"motion": motion,"length": len(motion),"text": text_data,}
                    new_name_list.append(name)
                    length_list.append(len(motion))
                    # print(count)
                    count += 1
                    # print(name)
            except:
                pass
        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = motion.shape[1]
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data[
            "text"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        # # padding
        # if m_length < self.max_motion_length:
        #     motion = np.concatenate(
        #         [
        #             motion,
        #             np.zeros((self.max_motion_length - m_length, motion.shape[1])),
        #         ],
        #         axis=0,
        #     )
        # print(word_embeddings.shape, motion.shape, m_length)
        # print(tokens)

        # debug check nan
        if np.any(np.isnan(motion)):
            raise ValueError("nan in motion")

        return (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            "_".join(tokens),
        )
        # return caption, motion, m_length


class MotionFixRetarget(data.Dataset):

    def __init__(
        self,
        mean,
        std,
        split_file,
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        motion_dir,
        text_dir,
        tiny=False,
        debug=False,
        progress_bar=True,
        **kwargs,
    ):
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        # min_motion_len = 40 if dataset_name =='t2m' else 24
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length

        data_dict = {}
        id_list = []
        split_file_path = split_file.replace(".txt", "_motionfix.txt")
        with cs.open(split_file_path, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        self.id_list = id_list

        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10

        if progress_bar:
            enumerator = enumerate(track(id_list,f"Loading MotionFix Retarget {split_file.split('/')[-1].split('.')[0]}",))
        else:
            enumerator = enumerate(id_list)
        count = 0
        bad_count = 0
        new_name_list = []
        length_list = []
        for i, name in enumerator:
            if count > maxdata:
                break
            try:
                motion_data = np.load(pjoin(motion_dir, name + ".npy"), allow_pickle=True).item()
                source = motion_data["source"]
                target = motion_data["target"]
                if (len(source)) < self.min_motion_length or (len(source) >=200):
                    bad_count += 1
                    continue
                if (len(target)) < self.min_motion_length or (len(target) >=200):
                    bad_count += 1
                    continue

                text_data = []
                with cs.open(pjoin(text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_data.append(line.strip())
                    flag = True

                if flag:
                    data_dict[name] = {"source": source, "target":target, "length_source": len(source), "length_target":len(target), "text": text_data,}
                    new_name_list.append(name)
                    length_list.append(len(target))
                    count += 1
            except:
                pass
        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = target.shape[1]
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        source, target, length_source, length_target, caption = data["source"], data["target"], data["length_source"], data["length_target"], data["text"][0]
        # Randomly select a caption

        "Z Normalization"
        source = (source - self.mean) / self.std
        target = (target - self.mean) / self.std

        # # padding
        # if m_length < self.max_motion_length:
        #     motion = np.concatenate(
        #         [
        #             motion,
        #             np.zeros((self.max_motion_length - m_length, motion.shape[1])),
        #         ],
        #         axis=0,
        #     )
        # print(word_embeddings.shape, motion.shape, m_length)
        # print(tokens)

        # debug check nan
        if np.any(np.isnan(source)):
            raise ValueError("nan in source")
        if np.any(np.isnan(target)):
            raise ValueError("nan in target")

        return (
            caption,
            source,
            target,
            length_source,
            length_target,
        )
        # return caption, motion, m_length

"""For use of training baseline"""


class Text2MotionDatasetBaseline(data.Dataset):

    def __init__(self, opt, mean, std, split_file, w_vectorizer):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name == "t2m" else 24

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + ".npy"))
                if (len(motion)) < min_motion_len or (len(motion) >= 200):
                    continue
                text_data = []
                flag = False
                with cs.open(pjoin(opt.text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                n_motion = motion[int(f_tag * 20):int(to_tag *
                                                                      20)]
                                if (len(n_motion)) < min_motion_len or (
                                        len(n_motion) >= 200):
                                    continue
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
                                    "_" + name)
                                while new_name in data_dict:
                                    new_name = (random.choice(
                                        "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
                                                name)
                                data_dict[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict],
                                }
                                new_name_list.append(new_name)
                                length_list.append(len(n_motion))
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                      to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data,
                    }
                    new_name_list.append(name)
                    length_list.append(len(motion))
            except:
                pass

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.nfeats = motion.shape[1]
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data[
            "text"]
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        len_gap = (m_length - self.max_length) // self.opt.unit_length

        if m_length != self.max_length:
            # print("Motion original length:%d_%d"%(m_length, len(motion)))
            if self.opt.unit_length < 10:
                coin2 = np.random.choice(["single", "single", "double"])
            else:
                coin2 = "single"
            if len_gap == 0 or (len_gap == 1 and coin2 == "double"):
                m_length = self.max_length
                s_idx = random.randint(0, m_length - self.max_length)
            else:
                if coin2 == "single":
                    n_m_length = self.max_length + self.opt.unit_length * len_gap
                else:
                    n_m_length = self.max_length + self.opt.unit_length * (
                        len_gap - 1)
                s_idx = random.randint(0, m_length - n_m_length)
                m_length = n_m_length
        else:
            s_idx = 0

        src_motion = motion[s_idx:s_idx + m_length]
        tgt_motion = motion[s_idx:s_idx + self.max_length]
        "Z Normalization"
        src_motion = (src_motion - self.mean) / self.std
        tgt_motion = (tgt_motion - self.mean) / self.std

        # padding
        if m_length < self.max_motion_length:
            src_motion = np.concatenate(
                [
                    src_motion,
                    np.zeros(
                        (self.max_motion_length - m_length, motion.shape[1])),
                ],
                axis=0,
            )
        # print(m_length, src_motion.shape, tgt_motion.shape)
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, caption, sent_len, src_motion, tgt_motion, m_length


class MotionDatasetV2(data.Dataset):

    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        joints_num = opt.joints_num

        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + ".npy"))
                if motion.shape[0] < opt.window_size:
                    continue
                self.lengths.append(motion.shape[0] - opt.window_size)
                self.data.append(motion)
            except:
                # Some motion may not exist in KIT dataset
                pass

        self.cumsum = np.cumsum([0] + self.lengths)

        if opt.is_train:
            # root_rot_velocity (B, seq_len, 1)
            std[0:1] = std[0:1] / opt.feat_bias
            # root_linear_velocity (B, seq_len, 2)
            std[1:3] = std[1:3] / opt.feat_bias
            # root_y (B, seq_len, 1)
            std[3:4] = std[3:4] / opt.feat_bias
            # ric_data (B, seq_len, (joint_num - 1)*3)
            std[4:4 + (joints_num - 1) * 3] = std[4:4 +
                                                  (joints_num - 1) * 3] / 1.0
            # rot_data (B, seq_len, (joint_num - 1)*6)
            std[4 + (joints_num - 1) * 3:4 +
                (joints_num - 1) * 9] = (std[4 + (joints_num - 1) * 3:4 +
                                             (joints_num - 1) * 9] / 1.0)
            # local_velocity (B, seq_len, joint_num*3)
            std[4 + (joints_num - 1) * 9:4 + (joints_num - 1) * 9 +
                joints_num * 3] = (std[4 + (joints_num - 1) * 9:4 +
                                       (joints_num - 1) * 9 + joints_num * 3] /
                                   1.0)
            # foot contact (B, seq_len, 4)
            std[4 + (joints_num - 1) * 9 + joints_num * 3:] = (
                std[4 +
                    (joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias)

            assert 4 + (joints_num -
                        1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
            np.save(pjoin(opt.meta_dir, "mean.npy"), mean)
            np.save(pjoin(opt.meta_dir, "std.npy"), std)

        self.mean = mean
        self.std = std
        print("Total number of motions {}, snippets {}".format(
            len(self.data), self.cumsum[-1]))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx + self.opt.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion


class RawTextDataset(data.Dataset):

    def __init__(self, opt, mean, std, text_file, w_vectorizer):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.nlp = spacy.load("en_core_web_sm")

        with cs.open(text_file) as f:
            for line in f.readlines():
                word_list, pos_list = self.process_text(line.strip())
                tokens = [
                    "%s/%s" % (word_list[i], pos_list[i])
                    for i in range(len(word_list))
                ]
                self.data_dict.append({
                    "caption": line.strip(),
                    "tokens": tokens
                })

        self.w_vectorizer = w_vectorizer
        print("Total number of descriptions {}".format(len(self.data_dict)))

    def process_text(self, sentence):
        sentence = sentence.replace("-", "")
        doc = self.nlp(sentence)
        word_list = []
        pos_list = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if (token.pos_ == "NOUN"
                    or token.pos_ == "VERB") and (word != "left"):
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[item]
        caption, tokens = data["caption"], data["tokens"]

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        return word_embeddings, pos_one_hots, caption, sent_len


class TextOnlyDataset(data.Dataset):

    def __init__(self, opt, mean, std, split_file, text_dir, **kwargs):
        self.mean = mean
        self.std = std
        self.opt = opt
        self.data_dict = []
        self.max_length = 20
        self.pointer = 0
        self.fixed_length = 120

        data_dict = {}
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:200]

        new_name_list = []
        length_list = []
        for name in tqdm(id_list):
            try:
                text_data = []
                flag = False
                with cs.open(pjoin(text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_dict = {}
                        line_split = line.strip().split("#")
                        caption = line_split[0]
                        tokens = line_split[1].split(" ")
                        f_tag = float(line_split[2])
                        to_tag = float(line_split[3])
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag

                        text_dict["caption"] = caption
                        text_dict["tokens"] = tokens
                        if f_tag == 0.0 and to_tag == 0.0:
                            flag = True
                            text_data.append(text_dict)
                        else:
                            try:
                                new_name = (
                                    random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
                                    "_" + name)
                                while new_name in data_dict:
                                    new_name = (random.choice(
                                        "ABCDEFGHIJKLMNOPQRSTUVW") + "_" +
                                                name)
                                data_dict[new_name] = {"text": [text_dict]}
                                new_name_list.append(new_name)
                            except:
                                print(line_split)
                                print(line_split[2], line_split[3], f_tag,
                                      to_tag, name)
                                # break

                if flag:
                    data_dict[name] = {"text": text_data}
                    new_name_list.append(name)
            except:
                pass

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = new_name_list

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        text_list = data["text"]

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data["caption"], text_data["tokens"]
        return None, None, caption, None, np.array([0
                                                    ]), self.fixed_length, None
        # fixed_length can be set from outside before sampling


# A wrapper class for t2m original dataset for MDM purposes
class HumanML3D(data.Dataset):

    def __init__(self,
                 mode,
                 datapath="./dataset/humanml_opt.txt",
                 split="train",
                 **kwargs):
        self.mode = mode

        self.dataset_name = "t2m"
        self.dataname = "t2m"

        # Configurations of T2M dataset and KIT dataset is almost the same
        abs_base_path = f"."
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = (
            None  # torch.device('cuda:4') # This param is not in use in this context
        )
        opt = get_opt(dataset_opt_path, device)
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        opt.motion_dir = pjoin(abs_base_path, opt.motion_dir)
        opt.text_dir = pjoin(abs_base_path, opt.text_dir)
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.data_root = pjoin(abs_base_path, opt.data_root)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        self.opt = opt
        print("Loading dataset %s ..." % opt.dataset_name)

        if mode == "gt":
            # used by T2M models (including evaluators)
            self.mean = np.load(pjoin(opt.meta_dir, "mean.npy"))
            self.std = np.load(pjoin(opt.meta_dir, "std.npy"))
        elif mode in ["train", "eval", "text_only"]:
            # used by our models
            self.mean = np.load(pjoin(opt.data_root, "Mean.npy"))
            self.std = np.load(pjoin(opt.data_root, "Std.npy"))

        if mode == "eval":
            # used by T2M models (including evaluators)
            # this is to translate their norms to ours
            self.mean_for_eval = np.load(pjoin(opt.meta_dir, "mean.npy"))
            self.std_for_eval = np.load(pjoin(opt.meta_dir, "std.npy"))

        self.split_file = pjoin(opt.data_root, f"{split}.txt")
        if mode == "text_only":
            self.t2m_dataset = TextOnlyDataset(self.opt, self.mean, self.std,
                                               self.split_file)
        else:
            self.w_vectorizer = WordVectorizer(pjoin(abs_base_path, "glove"),
                                               "our_vab")
            self.t2m_dataset = Text2MotionDatasetV2(self.opt, self.mean,
                                                    self.std, self.split_file,
                                                    self.w_vectorizer)
            self.num_actions = 1  # dummy placeholder

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()


# A wrapper class for t2m original dataset for MDM purposes
class KIT(HumanML3D):

    def __init__(self,
                 mode,
                 datapath="./dataset/kit_opt.txt",
                 split="train",
                 **kwargs):
        super(KIT, self).__init__(mode, datapath, split, **kwargs)


class Text2MotionDatasetCMLDTest(data.Dataset):

    def __init__(
        self,
        mean,
        std,
        split_file,
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        motion_dir,
        text_dir,
        tiny=False,
        debug=False,
        progress_bar=True,
        **kwargs,
    ):
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        # min_motion_len = 40 if dataset_name =='t2m' else 24
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length

        data_dict_1 = {}
        id_list_1 = []

        data_dict_2 = {}
        id_list_2 = []

        split_dir = os.path.dirname(split_file)
        split_base = os.path.basename(split_file).split(".")[0]
        split_subfile_1 = os.path.join(split_dir,split_base + "_humanml.txt")#_humanml
        split_subfile_2 = os.path.join(split_dir,split_base + "_100style_Filter.txt")#_100STYLE_Filter

        dict_path = "./datasets/humanml3d_100style/100style_name_dict_Filter.txt"
        motion_to_label = self.build_dict_from_txt(dict_path)
        motion_to_style_text = self.build_dict_from_txt(dict_path,is_style_text=True)

        with cs.open(split_subfile_1, "r") as f:
            for line in f.readlines():
                id_list_1.append(line.strip())
        
        num = len(id_list_1)
        
        random_samples = np.random.choice(range(num), size=100, replace=False)
        id_list_1 = np.array(id_list_1)
        # Use random_samples to index id_list_1_np
        # id_list_1 = id_list_1[random_samples]

        self.id_list_1 = id_list_1

        with cs.open(split_subfile_2, "r") as f:
            for line in f.readlines():
                id_list_2.append(line.strip())
        
        id_list_2 = id_list_2
        self.id_list_2 = id_list_2

        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10

        if progress_bar:
            enumerator_1 = enumerate(track(id_list_1,f"Loading HumanML3D {split_subfile_1.split('/')[-1].split('.')[0]}",))
        else:
            enumerator_1 = enumerate(id_list_1)

        count = 0
        bad_count = 0
        new_name_list_1 = []
        length_list_1 = []

        for i, name in enumerator_1:
            if count > maxdata:
                break
            motion = np.load(pjoin(motion_dir, name + ".npy"))
            if (len(motion)) < self.min_motion_length or (len(motion) >=200):
                bad_count += 1
                continue
            text_data_1 = []
            flag = False

            # name = style_to_neutral[name]
            text_path = pjoin(text_dir, name + ".txt")
            assert os.path.exists(text_path)
            with cs.open(text_path) as f:
                for line in f.readlines():
                    text_dict_1 = {}
                    line_split = line.strip().split("#")

                    caption = line_split[0]
                    tokens = line_split[1].split(" ")
                    f_tag = float(line_split[2])
                    to_tag = float(line_split[3])
                    f_tag = 0.0 if np.isnan(f_tag) else f_tag
                    to_tag = 0.0 if np.isnan(to_tag) else to_tag

                    text_dict_1["caption"] = caption
                    text_dict_1["tokens"] = tokens
                    text_data_1.append(text_dict_1)
                    if f_tag == 0.0 and to_tag == 0.0:
                        flag = True
                        text_data_1.append(text_dict_1)
                    else:
                        try:
                            n_motion = motion[int(f_tag * 20):int(to_tag *20)]
                            if (len(n_motion)) < self.min_motion_length or ((len(n_motion) >= 200)):
                                continue
                            new_name = (
                                random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
                                "_" + name)
                            while new_name in data_dict_1:
                                new_name = (random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" +name)
                            data_dict_1[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict_1],
                                }
                            new_name_list_1.append(new_name)
                            length_list_1.append(len(n_motion))
                        except:
                            print(line_split)
                            print(line_split[2], line_split[3], f_tag,to_tag, name)

                
                if flag:
                    data_dict_1[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data_1,
                    }
                    new_name_list_1.append(name)
                    length_list_1.append(len(motion))
                    count += 1            

        name_list_1, length_list_1 = zip(*sorted(zip(new_name_list_1, length_list_1), key=lambda x: x[1]))

        if progress_bar:
            enumerator_2 = enumerate(
                track(
                    id_list_2,
                    f"Loading 100STYLE {split_subfile_2.split('/')[-1].split('.')[0]}",
                ))
        else:
            enumerator_2 = enumerate(id_list_2)

        count = 0
        bad_count = 0
        new_name_list_2 = []
        length_list_2 = []

        for i, name in enumerator_2:
            if count > maxdata:
                break
            motion = np.load(pjoin(motion_dir, name + ".npy"))
            label_data = motion_to_label[name]
            style_text = motion_to_style_text[name]

            if (len(motion)) < self.min_motion_length or (len(motion) >=200):
                bad_count += 1
                continue
            text_data_2 = []
            flag = True

            text_path = pjoin(text_dir, name + ".txt")
            assert os.path.exists(text_path)
            with cs.open(text_path) as f:
                for line in f.readlines():
                    text_dict_2 = {}
                    line_split = line.strip().split("#")
                    caption = line_split[0]
                    text_dict_2["caption"] = caption
                    text_data_2.append(text_dict_2)

                if flag:
                    data_dict_2[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data_2,
                        "label": label_data,
                        "style_text":style_text,
                    }
                    new_name_list_2.append(name)
                    length_list_2.append(len(motion))
                    count += 1            

        name_list_2, length_list_2 = zip(*sorted(zip(new_name_list_2, length_list_2), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr_1 = np.array(length_list_1)
        self.data_dict_1 = data_dict_1
        self.name_list = name_list_1

        self.length_arr_2 = np.array(length_list_2)
        self.data_dict_2 = data_dict_2
        self.name_list_2 = name_list_2

        self.nfeats = motion.shape[1]
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length

        self.pointer_1 = np.searchsorted(self.length_arr_1, length)
        print("Pointer Pointing at %d" % self.pointer_1)

        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def transform(self,data):
        return (data - self.mean) / self.std

    def get_mean_std(self):
        return self.mean, self.std
    
    def __len__(self):
        return len(self.name_list) - self.pointer

    def build_dict_from_txt(filename,is_style=True,is_style_text=False):
        filename = "./datasets/humanml3d_100style/100style_name_dict_Filter.txt"
        result_dict = {}
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split(" ")
                if len(parts) >= 3:
                    key = parts[0]
                    if is_style and is_style_text == False:
                        value = parts[2]
                    elif is_style_text:
                        value = parts[1].split("_")[0]
                    else:
                        value = parts[3]


                    result_dict[key] = value
                    
        return result_dict

    def __getitem__(self, item):
        idx_1 = (self.pointer_1 + item) % len(self.name_list)
        data_1 = self.data_dict_1[self.name_list[idx_1]]
        motion_1, m_length_1, text_list_1 = data_1["motion"], data_1["length"], data_1["text"]

        idx_2 = (self.pointer_1 + item + random.randint(0,len(self.name_list_2)-1)) % len(self.name_list_2)
        data_2 = self.data_dict_2[self.name_list_2[idx_2]]
        motion_2, m_length_2, text_list_2,label,style_text = data_2["motion"], data_2["length"], data_2["text"], data_2["label"], data_2["style_text"]
      
        # Randomly select a caption
        text_data_1 = random.choice(text_list_1)
        caption_1,tokens = text_data_1["caption"], text_data_1["tokens"]

        text_data_2 = random.choice(text_list_2)
        caption_2 = text_data_2["caption"]

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)


        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length_1 = (m_length_1 // self.unit_length - 1) * self.unit_length
            m_length_2 = (m_length_2 // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length_1 = (m_length_1 // self.unit_length) * self.unit_length
            m_length_2 = (m_length_2 // self.unit_length) * self.unit_length
        
        idx_1 = random.randint(0, len(motion_1) - m_length_1)
        motion_1 = motion_1[idx_1:idx_1 + m_length_1]
        "Z Normalization"
        motion_1 = (motion_1 - self.mean) / self.std

        idx_2 = random.randint(0, len(motion_2) - m_length_2)
        motion_2 = motion_2[idx_2:idx_2 + m_length_2]
        "Z Normalization"
        motion_2 = (motion_2 - self.mean) / self.std

        # debug check nan
        if np.any(np.isnan(motion_1)):
            raise ValueError("nan in motion")

        return (
            word_embeddings,
            pos_one_hots,
            caption_1,
            sent_len,
            motion_1,
            m_length_1,
            "_".join(tokens),

            caption_2,
            motion_2,
            m_length_2,
            label,
            style_text,
        )
    

class AllDataset(data.Dataset):

    def __init__(
        self,
        mean,
        std,
        split_file,
        w_vectorizer,
        max_motion_length,
        min_motion_length,
        max_text_len,
        unit_length,
        motion_dir,
        text_dir,
        tiny=False,
        debug=False,
        progress_bar=True,
        **kwargs,
    ):
        self.w_vectorizer = w_vectorizer
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = max_motion_length
        # min_motion_len = 40 if dataset_name =='t2m' else 24
        self.min_motion_length = min_motion_length
        self.max_text_len = max_text_len
        self.unit_length = unit_length

        data_dict_1 = {}
        id_list_1 = []

        data_dict_2 = {}
        id_list_2 = []

        data_dict_3 = {}
        id_list_3 = []

        split_dir = os.path.dirname(split_file)
        split_base = os.path.basename(split_file).split(".")[0]
        split_subfile_1 = os.path.join(split_dir,split_base + "_humanml.txt")
        # split_subfile_2 = os.path.join(split_dir,split_base + "_100style_Filter.txt")
        split_subfile_3 = os.path.join(split_dir,split_base + "_motionfix.txt")

        # dict_path = "./datasets/humanml3d_100style/100style_name_dict_Filter.txt"
        # motion_to_label = self.build_dict_from_txt(dict_path)
        # motion_to_style_text = self.build_dict_from_txt(dict_path,is_style_text=True)

        ## Load the HumanML3D dataset
        with cs.open(split_subfile_1, "r") as f:
            for line in f.readlines():
                id_list_1.append(line.strip())
        id_list_1 = np.array(id_list_1)
        self.id_list_1 = id_list_1

        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10

        if progress_bar:
            enumerator_1 = enumerate(track(id_list_1,f"Loading HumanML3D {split_subfile_1.split('/')[-1].split('.')[0]}",))
        else:
            enumerator_1 = enumerate(id_list_1)

        count = 0
        bad_count = 0
        new_name_list_1 = []
        length_list_1 = []

        for i, name in enumerator_1:
            if count > maxdata:
                break
            motion = np.load(pjoin(motion_dir, name + ".npy"))
            if (len(motion)) < self.min_motion_length or (len(motion) >=200):
                bad_count += 1
                continue
            text_data_1 = []
            flag = False

            # name = style_to_neutral[name]
            text_path = pjoin(text_dir, name + ".txt")
            assert os.path.exists(text_path)
            with cs.open(text_path) as f:
                for line in f.readlines():
                    text_dict_1 = {}
                    line_split = line.strip().split("#")

                    caption = line_split[0]
                    tokens = line_split[1].split(" ")
                    f_tag = float(line_split[2])
                    to_tag = float(line_split[3])
                    f_tag = 0.0 if np.isnan(f_tag) else f_tag
                    to_tag = 0.0 if np.isnan(to_tag) else to_tag

                    text_dict_1["caption"] = caption
                    text_dict_1["tokens"] = tokens
                    text_data_1.append(text_dict_1)
                    if f_tag == 0.0 and to_tag == 0.0:
                        flag = True
                        text_data_1.append(text_dict_1)
                    else:
                        try:
                            n_motion = motion[int(f_tag * 20):int(to_tag *20)]
                            if (len(n_motion)) < self.min_motion_length or ((len(n_motion) >= 200)):
                                continue
                            new_name = (
                                random.choice("ABCDEFGHIJKLMNOPQRSTUVW") +
                                "_" + name)
                            while new_name in data_dict_1:
                                new_name = (random.choice("ABCDEFGHIJKLMNOPQRSTUVW") + "_" +name)
                            data_dict_1[new_name] = {
                                    "motion": n_motion,
                                    "length": len(n_motion),
                                    "text": [text_dict_1],
                                }
                            new_name_list_1.append(new_name)
                            length_list_1.append(len(n_motion))
                        except:
                            print(line_split)
                            print(line_split[2], line_split[3], f_tag,to_tag, name)

                
                if flag:
                    data_dict_1[name] = {
                        "motion": motion,
                        "length": len(motion),
                        "text": text_data_1,
                    }
                    new_name_list_1.append(name)
                    length_list_1.append(len(motion))
                    count += 1            

        name_list_1, length_list_1 = zip(*sorted(zip(new_name_list_1, length_list_1), key=lambda x: x[1]))

        # ## Load the 100STYLE dataset
        # with cs.open(split_subfile_2, "r") as f:
        #     for line in f.readlines():
        #         id_list_2.append(line.strip())
        
        # id_list_2 = id_list_2
        # self.id_list_2 = id_list_2

        # if progress_bar:
        #     enumerator_2 = enumerate(
        #         track(
        #             id_list_2,
        #             f"Loading 100STYLE {split_subfile_2.split('/')[-1].split('.')[0]}",
        #         ))
        # else:
        #     enumerator_2 = enumerate(id_list_2)

        # count = 0
        # bad_count = 0
        # new_name_list_2 = []
        # length_list_2 = []

        # for i, name in enumerator_2:
        #     if count > maxdata:
        #         break
        #     motion = np.load(pjoin(motion_dir, name + ".npy"))
        #     label_data = motion_to_label[name]
        #     style_text = motion_to_style_text[name]

        #     if (len(motion)) < self.min_motion_length or (len(motion) >=200):
        #         bad_count += 1
        #         continue
        #     text_data_2 = []
        #     flag = True

        #     text_path = pjoin(text_dir, name + ".txt")
        #     assert os.path.exists(text_path)
        #     with cs.open(text_path) as f:
        #         for line in f.readlines():
        #             text_dict_2 = {}
        #             line_split = line.strip().split("#")
        #             caption = line_split[0]
        #             text_dict_2["caption"] = caption
        #             text_data_2.append(text_dict_2)

        #         if flag:
        #             data_dict_2[name] = {
        #                 "motion": motion,
        #                 "length": len(motion),
        #                 "text": text_data_2,
        #                 "label": label_data,
        #                 "style_text":style_text,
        #             }
        #             new_name_list_2.append(name)
        #             length_list_2.append(len(motion))
        #             count += 1            

        # name_list_2, length_list_2 = zip(*sorted(zip(new_name_list_2, length_list_2), key=lambda x: x[1]))

        ## Load the MotionFix dataset
        with cs.open(split_subfile_3, "r") as f:
            for line in f.readlines():
                id_list_3.append(line.strip())
        self.id_list_3 = id_list_3

        if tiny or debug:
            progress_bar = False
            maxdata = 10 if tiny else 100
        else:
            maxdata = 1e10

        if progress_bar:
            enumerator_3 = enumerate(track(id_list_3,f"Loading MotionFix Retarget {split_file.split('/')[-1].split('.')[0]}",))
        else:
            enumerator_3 = enumerate(id_list_3)
        count = 0
        bad_count = 0
        new_name_list_3 = []
        length_list_3 = []
        for i, name in enumerator_3:
            if count > maxdata:
                break
            try:
                motion_data = np.load(pjoin(motion_dir, name + ".npy"), allow_pickle=True).item()
                source = motion_data["source"]
                target = motion_data["target"]
                if (len(source)) < self.min_motion_length or (len(source) >=200):
                    bad_count += 1
                    continue
                if (len(target)) < self.min_motion_length or (len(target) >=200):
                    bad_count += 1
                    continue

                text_data_3 = []
                with cs.open(pjoin(text_dir, name + ".txt")) as f:
                    for line in f.readlines():
                        text_data_3.append(line.strip())
                    flag = True

                if flag:
                    data_dict_3[name] = {"source": source, "target":target, "length_source": len(source), "length_target":len(target), "text": text_data_3,}
                    new_name_list_3.append(name)
                    length_list_3.append(len(target))
                    count += 1
            except:
                pass
        name_list_3, length_list_3 = zip(*sorted(zip(new_name_list_3, length_list_3), key=lambda x: x[1]))


        self.mean = mean
        self.std = std
        self.length_arr_1 = np.array(length_list_1)
        self.data_dict_1 = data_dict_1
        self.name_list = name_list_1

        # self.length_arr_2 = np.array(length_list_2)
        # self.data_dict_2 = data_dict_2
        # self.name_list_2 = name_list_2

        self.length_arr_3 = np.array(length_list_3)
        self.data_dict_3 = data_dict_3
        self.name_list_3 = name_list_3

        self.nfeats = motion.shape[1]
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length

        self.pointer_1 = np.searchsorted(self.length_arr_1, length)
        print("Pointer Pointing at %d" % self.pointer_1)

        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def transform(self,data):
        return (data - self.mean) / self.std

    def get_mean_std(self):
        return self.mean, self.std
    
    def __len__(self):
        return len(self.name_list) - self.pointer

    def build_dict_from_txt(filename,is_style=True,is_style_text=False):
        filename = "./datasets/humanml3d_100style/100style_name_dict_Filter.txt"
        result_dict = {}
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split(" ")
                if len(parts) >= 3:
                    key = parts[0]
                    if is_style and is_style_text == False:
                        value = parts[2]
                    elif is_style_text:
                        value = parts[1].split("_")[0]
                    else:
                        value = parts[3]


                    result_dict[key] = value
                    
        return result_dict

    def __getitem__(self, item):
        idx_1 = (self.pointer_1 + item) % len(self.name_list)
        data_1 = self.data_dict_1[self.name_list[idx_1]]
        motion_1, m_length_1, text_list_1 = data_1["motion"], data_1["length"], data_1["text"]

        # idx_2 = (self.pointer_1 + item + random.randint(0,len(self.name_list_2)-1)) % len(self.name_list_2)
        # data_2 = self.data_dict_2[self.name_list_2[idx_2]]
        # motion_2, m_length_2, text_list_2,label,style_text = data_2["motion"], data_2["length"], data_2["text"], data_2["label"], data_2["style_text"]
      
        # Randomly select a caption
        text_data_1 = random.choice(text_list_1)
        caption_1,tokens = text_data_1["caption"], text_data_1["tokens"]

        # text_data_2 = random.choice(text_list_2)
        # caption_2 = text_data_2["caption"]

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"
                               ] * (self.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)


        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length_1 = (m_length_1 // self.unit_length - 1) * self.unit_length
            # m_length_2 = (m_length_2 // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length_1 = (m_length_1 // self.unit_length) * self.unit_length
            # m_length_2 = (m_length_2 // self.unit_length) * self.unit_length
        
        idx_1 = random.randint(0, len(motion_1) - m_length_1)
        motion_1 = motion_1[idx_1:idx_1 + m_length_1]
        "Z Normalization"
        motion_1 = (motion_1 - self.mean) / self.std

        # idx_2 = random.randint(0, len(motion_2) - m_length_2)
        # motion_2 = motion_2[idx_2:idx_2 + m_length_2]
        # "Z Normalization"
        # motion_2 = (motion_2 - self.mean) / self.std

        # debug check nan
        if np.any(np.isnan(motion_1)):
            raise ValueError("nan in motion")
        
        
        idx_3 = (self.pointer_1 + item + random.randint(0,len(self.name_list_3)-1)) % len(self.name_list_3)
        data_3 = self.data_dict_3[self.name_list_3[idx_3]]
        source, target, length_source, length_target, caption_3 = data_3["source"], data_3["target"], data_3["length_source"], data_3["length_target"], data_3["text"][0]

        crop_proportion = random.uniform(0.5, 1)
        crop_start = random.uniform(0, 1 - crop_proportion)

        length = math.ceil(source.shape[0] * crop_proportion)
        start_index = math.floor(source.shape[0] * crop_start)
        source = source[start_index:start_index + length]  
        length_source = length

        length = math.ceil(target.shape[0] * crop_proportion)
        start_index = math.floor(target.shape[0] * crop_start)
        target = target[start_index:start_index + length]
        length_target = length

        "Z Normalization"
        source = (source - self.mean) / self.std
        target = (target - self.mean) / self.std

        # # padding
        # if m_length < self.max_motion_length:
        #     motion = np.concatenate(
        #         [
        #             motion,
        #             np.zeros((self.max_motion_length - m_length, motion.shape[1])),
        #         ],
        #         axis=0,
        #     )
        # print(word_embeddings.shape, motion.shape, m_length)
        # print(tokens)

        # debug check nan
        if np.any(np.isnan(source)):
            raise ValueError("nan in source")
        if np.any(np.isnan(target)):
            raise ValueError("nan in target")

        return (
            word_embeddings,
            pos_one_hots,
            caption_1,
            sent_len,
            motion_1,
            m_length_1,
            "_".join(tokens),

            # caption_2,
            # motion_2,
            # m_length_2,
            # label,
            # style_text,

            caption_3,
            source,
            target,
            length_source,
            length_target,
        )
