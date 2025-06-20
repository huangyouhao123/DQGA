import os

import gym
import numpy as np

from entity.resource import *
from entity.patients import *
from util.config import config
from util.util import parse_args, copy_data_folder_to_output, build_network


def M1_std(m1: np.ndarray):
    return m1 / m1.max() if m1.max() > 0 else m1


def M2_std(m1: np.ndarray):
    return m1 / m1.max() if m1.max() > 0 else m1


class Env:
    def __init__(self, args, resource, patients):
        self.patients = None
        self.After_sources = None
        self.Before_sources = None
        self.Operation_rooms = None
        self.Doctors = None
        self.args = args
        self.before_num = resource['before']
        self.after_num = resource['after']
        self.OR_types = [chr(ord('A') + i) for i in range(resource['ORtype'])]
        self.source_nums = [resource[tp][0] for tp in self.OR_types]
        self.doctor_nums = [resource[tp][2] for tp in self.OR_types]
        self.source_nums = [self.before_num] + self.source_nums + [self.after_num]
        self.num_of_all_source = sum(self.source_nums)
        self.type_patients_num = {OR: len([pt for pt in patients if pt[0] == OR]) for OR in self.OR_types}
        self.num_of_all_patients = len(patients)
        self.patients_data = patients
        self.source_idx = []
        start = 0
        for source_num in self.source_nums:
            self.source_idx.append([start + bias for bias in range(source_num)])
            start += source_num

        self.patients_dict = {i: [] for i in self.OR_types}
        self.init_items()
        # self.Matrix1 = None
        # self.Matrix2 = None
        # self.Matrix3 = None
        self.finished = []
        self.Num_finished = 0
        self.cmax = 0
        self.c_max_dict = {OR: 0 for OR in self.OR_types}
        self.important_weight = {OR: resource[OR][1] for OR in self.OR_types}  # 不同科室的cmax权重占比
        self.eta = {OR: resource[OR][3] for OR in self.OR_types}
        self.award_weight = [0.7, 0.3, 0.5]
        self.all_processing_time = 0
        self.department_processing_time = {OR: 0 for OR in self.OR_types}
        self.u = 0
        self.max_time = None

    def init_items(self):
        self.Operation_rooms = []
        self.Doctors = []
        for idx, type in enumerate(self.OR_types):
            for i in range(self.source_nums[idx + 1]):
                self.Operation_rooms.append(OperationRoom(type, i))
        for idx, type in enumerate(self.OR_types):
            for i in range(self.doctor_nums[idx]):
                self.Doctors.append(Doctor(type, i))
        self.Before_sources = [Before_resource(i) for i in range(self.before_num)]
        self.After_sources = [After_resource(i) for i in range(self.after_num)]
        self.patients = [Patient(idx, p[0], p[1], p[2], p[3], p[4]) for idx, p in enumerate(self.patients_data)]
        for idx,doc in enumerate(self.Doctors):
            doc.set_abs_idx(idx)
        for idx, resource in enumerate(self.Before_sources + self.Operation_rooms + self.After_sources):
            resource.absolute_idx = idx
        self.patients_dict = {i: [] for i in self.OR_types}
        for idx, patient in enumerate(self.patients):
            patient.idx_list = self.source_idx
            patient.stage_idx = patient.idx_list[0]
            patient.doctor = self.Doctors[self.change_idx(patient.type, patient.doctor_idx)]
            self.patients_dict[patient.type].append(idx)

    def change_idx(self, ORtype, idx):
        count = 0
        ans = 0
        while count + ord('A') < ord(ORtype):
            ans += self.doctor_nums[count]
            count += 1
        return ans + idx

    def init_start_matrix(self):
        self.start_Matrix = []

        for patient in self.patients:
            patient_vector = []
            avg_before = patient.before_cost / self.source_nums[0]
            patient_vector = patient_vector + [avg_before for _ in range(self.source_nums[0])]
            for idx, num in enumerate(self.source_nums[1:-1]):
                if patient.type == chr(idx + ord('A')):
                    avg_before = patient.OR_cost / num
                    patient_vector = patient_vector + [avg_before for _ in range(num)]
                else:
                    patient_vector = patient_vector + [0 for _ in range(num)]
            avg_after = patient.after_cost / self.source_nums[-1]
            patient_vector += [avg_after for _ in range(self.source_nums[-1])]
            self.start_Matrix.append(patient_vector)
            patient.idx_list = self.source_idx
            patient.stage_idx = patient.idx_list[0]
        self.start_Matrix = np.array(self.start_Matrix)
        self.max_time = self.start_Matrix.sum()

    def C_max(self):
        m = 0
        for Mi in self.Before_sources:
            if Mi.end > m:
                m = Mi.end
        for Mi in self.After_sources:
            if Mi.end > m:
                m = Mi.end
        for Mi in self.Operation_rooms:
            if Mi.end > m:
                m = Mi.end
        return m

    def avg_start(self):
        self.Matrix3: np.ndarray
        ans = (self.s[2].sum() - (
                self.max_time * self.num_of_all_patients * (self.num_of_all_source - 3))) / (
                      self.num_of_all_patients * 3)
        return ans

    def cost_time_award(self):
        cta = 0
        self.update_cmax_dict()
        self.department_processing_time = {OR: 0 for OR in self.OR_types}
        for pt in self.patients:
            tm = pt.before_cost + pt.OR_cost + pt.after_cost
            self.department_processing_time[pt.type] += tm
        for OR in self.OR_types:
            cta += self.important_weight[OR] * self.department_processing_time[OR] / self.c_max_dict[OR] / \
                   self.type_patients_num[OR] if self.c_max_dict[OR] != 0 else 0
        return cta

    def move_with_penalty(self, pt: Patient, resources_idx, unfit_list, ufi: list):
        resources = self.Before_sources + self.Operation_rooms + self.After_sources
        resource = resources[resources_idx]
        to_s = pt.start_time[1] - pt.before_cost
        to_e = pt.start_time[1]

        # 标记已经处理的病人，防止重复处理
        if pt.idx not in ufi:
            return True

        for idx, pt_idx in enumerate(resource._on):
            if pt_idx == pt.idx:
                if idx == len(resource._on) - 1:
                    pt.start_time[0] = to_s
                    pt.finish_time[0] = to_e
                    resource.start[idx] = to_s
                    resource.finish[idx] = to_e
                    pt.begin = to_s
                    ufi.remove(pt.idx)
                    return True
                else:
                    next_patient = self.patients[resource._on[idx + 1]]

                    # 避免循环递归，防止回路问题
                    if next_patient.idx in ufi:
                        if self.move_with_penalty(next_patient, resources_idx, unfit_list, ufi):
                            if to_e <= next_patient.start_time[0]:
                                pt.start_time[0] = to_s
                                pt.finish_time[0] = to_e
                                resource.start[idx] = to_s
                                resource.finish[idx] = to_e
                                pt.begin = to_s
                                ufi.remove(pt.idx)
                                return True
                        else:
                            return False
                    else:
                        return False

        return False

    def move_with_penalty2(self, pt: Patient, resources_idx, unfit_list, ufi: list):
        resources = self.Before_sources + self.Operation_rooms + self.After_sources
        resource = resources[resources_idx]
        to_s = pt.start_time[2] - pt.OR_cost
        to_e = pt.start_time[2]

        # 标记已经处理的病人，防止重复处理
        if pt.idx not in ufi:
            return True

        for idx, pt_idx in enumerate(resource._on):
            if pt_idx == pt.idx:
                if idx == len(resource._on) - 1:
                    pt.start_time[1] = to_s
                    pt.finish_time[1] = to_e
                    resource.start[idx] = to_s
                    resource.finish[idx] = to_e
                    pt.begin = to_s
                    ufi.remove(pt.idx)
                    return True
                else:
                    next_patient = self.patients[resource._on[idx + 1]]

                    # 避免循环递归，防止回路问题
                    if next_patient.idx in ufi:
                        if self.move_with_penalty(next_patient, resources_idx, unfit_list, ufi):
                            if to_e <= next_patient.start_time[0]:
                                pt.start_time[1] = to_s
                                pt.finish_time[1] = to_e
                                resource.start[idx] = to_s
                                resource.finish[idx] = to_e
                                pt.begin = to_s
                                ufi.remove(pt.idx)
                                return True
                        else:
                            return False
                    else:
                        return False

        return False
    def modify_patients2(self,not_modify_patients):
        unfit_list = []
        unfit_list_idx = []
        for pt in self.patients:
            pt: Patient
            if len(pt.start_time)>2 and len(pt.finish_time)>1 and pt.start_time[2] != pt.finish_time[1]:
                unfit_list.append(pt)
                unfit_list_idx.append(pt.idx)
        if not_modify_patients is not None and len(not_modify_patients) > 0:
            for pt_idx in not_modify_patients:
                if self.patients[pt_idx] in unfit_list:
                    unfit_list.remove(self.patients[pt_idx])
                    unfit_list_idx.remove(self.patients[pt_idx].idx)
        for pt in unfit_list:
            self.move_with_penalty2(pt, pt.on[1], unfit_list, unfit_list_idx)
    def modify_patients(self,not_modify_patients):
        unfit_list = []
        unfit_list_idx = []
        for pt in self.patients:
            pt: Patient
            if len(pt.start_time)>1 and len(pt.finish_time)>0 and pt.start_time[1] != pt.finish_time[0]:
                unfit_list.append(pt)
                unfit_list_idx.append(pt.idx)
        if not_modify_patients is not None and len(not_modify_patients) > 0:
            for pt_idx in not_modify_patients:
                if self.patients[pt_idx] in unfit_list:
                    unfit_list.remove(self.patients[pt_idx])
                    unfit_list_idx.remove(self.patients[pt_idx].idx)
        for pt in unfit_list:
            self.move_with_penalty(pt, pt.on[0], unfit_list, unfit_list_idx)

    def waiting_time_award(self):
        wta = 0
        pt_dict = {i: [] for i in self.OR_types}
        self.department_processing_time = {OR: 0 for OR in self.OR_types}
        for pt in self.patients:
            tm = pt.before_cost + pt.OR_cost + pt.after_cost
            self.department_processing_time[pt.type] += tm
        for pat in self.patients:
            pt_dict[pat.type].append(pat.start_time[0])
        for OR in self.OR_types:
            avg_start = sum(pt_dict[OR]) / len(pt_dict[OR])
            wta += self.eta[OR] * (1 - avg_start / self.department_processing_time[OR])
            # print(OR,'alpha_s',1 - avg_start / self.department_processing_time[OR])
        return wta

    def gap_penalty(self):
        delta_T = 0
        for pt in self.patients:
            delta_T = delta_T + pt.start_time[1] - pt.finish_time[0] + pt.start_time[2] - pt.finish_time[1]
        return delta_T / self.C_max()

    def U(self,not_modify=None):
        if self.args.independent:
            without_m=self.cost_time_award() * self.award_weight[0] + self.waiting_time_award() * self.award_weight[
                1] - self.gap_penalty() * self.award_weight[2]
            self.modify_patients2(not_modify)
            self.modify_patients(not_modify)
            m=self.cost_time_award() * self.award_weight[0] + self.waiting_time_award() * self.award_weight[
                1] - self.gap_penalty() * self.award_weight[2]
            # print(self.cost_time_award(),self.waiting_time_award(),self.gap_penalty())
            # if not self.modify_patients():
            #     penalty=0.5
            # return (self.cost_time_award() * self.award_weight[0] + self.waiting_time_award() * self.award_weight[1])*penalty
            # print(self.cost_time_award() , self.waiting_time_award(), self.gap_penalty())
            return max(m, without_m)
        else:
            C_max = self.cmax
            # return self.P/(self.m*C_max)
            # return (1 / C_max + 1 / self.avg_start())*self.all_processing_time
            return self.award_weight[
                0] * self.all_processing_time / self.num_of_all_patients / C_max + self.waiting_time_award() * \
                self.award_weight[1]

    def reset(self):
        done = 0
        self.u = 0
        self.cmax = 0
        self.c_max_dict = {OR: 0 for OR in self.OR_types}
        self.department_processing_time = {OR: 0 for OR in self.OR_types}
        self.all_processing_time = 0
        self.finished = []
        self.Num_finished = 0
        self.init_items()
        for pt in self.patients:
            pt.begin = self.max_time
        self.start_Matrix: np.ndarray
        Matrix1 = self.start_Matrix.copy()  # 所剩时间
        Matrix2 = np.zeros_like(Matrix1)  # 资源安排
        Matrix3 = np.ones_like(Matrix1) * self.max_time  # 开始时间
        Matrix4 = np.zeros_like(Matrix1)
        self.s = np.stack((Matrix1, Matrix2, Matrix3, Matrix4), 0)
        matrix1 = M1_std(self.s[0])
        matrix2 = M2_std(self.s[1])
        # matrix3 = self.M3_std(self.s[2])
        matrix4 = self.m4_std(self.s[3])
        st = np.stack((matrix1), 0)
        if config.with_concrete_info:
            st = (st, None)
        else:
            st = (st, None)
        return st, done

    def M3_std(self, m1: np.ndarray):
        return m1 / self.max_time

    def m4_std(self, m4: np.ndarray):
        return m4 / self.cmax if self.cmax != 0 else m4

    def update_cmax_dict(self):
        for pat in self.patients:
            pat: Patient
            end_time = pat.end
            flag = pat.type
            self.c_max_dict[flag] = end_time if end_time > self.c_max_dict[flag] else self.c_max_dict[flag]

    def step(self, patient_idx, resource):
        self.start_Matrix: np.ndarray
        done = 0
        patient = self.patients[patient_idx]
        before_stage_idx = patient.stage_idx
        pt_type = patient.type
        pt = self.start_Matrix[patient_idx, patient.stage_idx].sum()
        resource.handling(patient, pt)
        self.department_processing_time[pt_type] += pt
        self.all_processing_time += pt
        self.s[0][patient_idx, before_stage_idx] = 0
        self.s[1][patient_idx, resource.absolute_idx] = self.start_Matrix[patient_idx, before_stage_idx].sum()
        self.s[2][patient_idx, resource.absolute_idx] = resource.start[-1]
        self.s[3][patient_idx, resource.absolute_idx] = resource.finish[-1]

        matrix1 = M1_std(self.s[0])
        st = np.stack((matrix1), 0)
        if patient.is_finished():
            self.finished.append(patient_idx)
            self.Num_finished += 1
        if self.Num_finished == self.num_of_all_patients:
            done = 1
        self.cmax = self.C_max()
        self.update_cmax_dict()

        # s=self.s.flatten()
        return st, done

