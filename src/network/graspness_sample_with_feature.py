from network.graspness_sample import GraspnessSample
import numpy as np
import torch


class GraspnessSampleWithFeature(GraspnessSample):
    def sample(
        self,
        data: dict,
        k: int,
        cate: bool = True,
        allow_fail: bool = False,
        graspness_scale=1,
        with_point=False,
        edge=None,
        with_graspness=False,
        ratio=0.005,
        near=False,
        with_score_parts=False,
        with_feature=False,
    ):
        pc_cuda = data['point_clouds']
        b = pc_cuda.shape[0]
        feature = self.get_feature(data)
        objectness, graspness = self.pred_score(feature)
        graspness = torch.where(
            objectness.argmax(dim=-1) == 1,
            graspness,
            torch.full_like(graspness, np.log(1e-3)),
        )
        if edge is not None:
            graspness = torch.where(
                edge == 0, graspness, torch.full_like(graspness, np.log(1e-3))
            )

        features = []
        seed_points = []
        graspnesses = []
        obj_indices = []

        for i in range(b):
            obj_indices.append([])
            if cate:
                seg = data['seg'][i]
                obj_ids = [idx for idx in torch.unique(seg).tolist() if idx != 0]
                obj_num = len(obj_ids)
                obj_k = [k // obj_num for _ in range(obj_num)]
                for _ in range(k % obj_num):
                    obj_k[np.random.randint(obj_num)] += 1

                for j, obj_id in enumerate(obj_ids):
                    graspable = (seg == obj_id).to(objectness.device)
                    graspness_obj = graspness[i, graspable].sort(descending=True).values
                    threshold = graspness_obj[int(graspness_obj.size(0) * 0.05)]
                    graspable = (seg == obj_id).to(objectness.device) & (
                        graspness[i] >= threshold
                    )
                    seed_point, indices = self.sample_points(pc_cuda[i], graspable, obj_k[j])
                    features.append(feature[i, graspable][indices][0])
                    seed_points.append(seed_point[0])
                    graspnesses.append(graspness[i, graspable][indices][0])
                    obj_indices[-1] += [obj_id] * obj_k[j]
            else:
                if near:
                    from pytorch3d.ops import ball_query
                    K = 1000
                    graspable = graspness[i] >= graspness[i].sort(descending=True).values[
                        int(graspness[i].size(0) * ratio)
                    ]
                    dists, idxs, nn = ball_query(
                        pc_cuda[i, graspable][None],
                        pc_cuda[i][None],
                        radius=0.02,
                        K=K,
                    )
                    graspness_around = torch.where(
                        idxs.reshape(-1) != -1,
                        graspness[i][idxs.reshape(-1)],
                        torch.full_like(idxs.reshape(-1), np.log(1e-3)).float(),
                    ).reshape(-1, K)
                    good = graspness_around > graspness_around.sort(descending=True).values[
                        torch.arange(len(idxs[0]), device=idxs.device),
                        ((idxs[0] != -1).sum(-1) * 0.1).long(),
                    ][:, None]
                    new_idxs = idxs.reshape(-1)[good.reshape(-1)]
                    if len(new_idxs) != 0:
                        graspable[:] = 0
                        graspable[new_idxs] = 1
                else:
                    graspable = graspness[i] >= graspness[i].sort(descending=True).values[
                        int(graspness[i].size(0) * ratio)
                    ]
                    graspable = graspness[i] > np.log(1e-2)
                    if graspable.sum() == 0:
                        graspable = graspness[i] >= graspness[i].sort(descending=True).values[
                            int(graspness[i].size(0) * ratio)
                        ]
                seed_point, indices = self.sample_points(pc_cuda[i], graspable, k)
                features.append(feature[i, graspable][indices][0])
                seed_points.append(seed_point[0])
                graspnesses.append(graspness[i, graspable][indices][0])
                obj_indices[-1] += [-1] * k

        features, seed_points = torch.cat(features), torch.cat(seed_points)
        rot, trans, joints, log_prob = self.sample_grasp(
            features, seed_points, sample_num=1, allow_fail=allow_fail
        )
        rot, trans, joints, log_prob = (
            rot.reshape(b, k, 3, 3),
            trans.reshape(b, k, 3),
            joints.reshape(b, k, -1),
            log_prob.reshape(b, k),
        )
        graspnesses = torch.cat(graspnesses, dim=0).reshape(b, k)

        score = log_prob + graspnesses * graspness_scale
        score = score.nan_to_num(nan=-1e6)

        obj_indices = torch.tensor(obj_indices).to(rot.device)

        result = [rot, trans, joints, score, obj_indices]
        if with_score_parts:
            result.append(graspnesses)
            result.append(log_prob)
        if with_point:
            result.append(seed_points)
        if with_graspness:
            result.append(graspable.float())
        if with_feature:
            result.append(features)
        return result
