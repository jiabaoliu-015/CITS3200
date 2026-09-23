# Perception model weights

These checkpoints are downloaded from the official [OpenPCDet model zoo](https://github.com/open-mmlab/OpenPCDet#model-zoo).
The dataset/configuration shown for each model must be respected when selecting
an evaluation dataset.

## VoxelNeXt (Argoverse 2)

- Source: [OpenPCDet model zoo Google Drive checkpoint](https://drive.google.com/uc?id=1YP2UOz-yO-cWfYQkIqILEu6bodvCBVrR)
- OpenPCDet config: `tools/cfgs/argo2_models/cbgs_voxel01_voxelnext.yaml`
- Training dataset: Argoverse 2
- Reported metric: 30.5 mAP
- File: `voxelnext_av2.pth`
- File size: 31,730,623 bytes
- SHA256: `dd123309fd6f196bd3df1bc54f02c0a26189c0109a25ff4bc875686467041705`

## PointPillar (KITTI)

- Source: [OpenPCDet model zoo Google Drive checkpoint](https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view?usp=sharing)
- OpenPCDet config: `tools/cfgs/kitti_models/pointpillar.yaml`
- Training/evaluation dataset: KITTI
- Reported metric: 77.28 Car@R11 on the KITTI validation set
- File: `pointpillar_kitti.pth`
- File size: 19,383,576 bytes
- SHA256: `4c83fc0fa02575b9b3e9dec676f698e7a70bb5a795e89f91df8a96b916fa19e2`

## SECOND (KITTI)

- Source: [OpenPCDet model zoo Google Drive checkpoint](https://drive.google.com/file/d/1-01zsPOsqanZQqIIyy7FpNXStL3y4jdR/view?usp=sharing)
- OpenPCDet config: `tools/cfgs/kitti_models/second.yaml`
- Training/evaluation dataset: KITTI
- Reported metric: 78.62 Car@R11 on the KITTI validation set
- File: `second_kitti.pth`
- File size: 21,355,695 bytes
- SHA256: `33e337b9ad2c2efad974f4827045551ffad272c7df3de8d44bee174f7e2d5136`

## Citation

```bibtex
@misc{openpcdet2020,
	title={OpenPCDet: An Open-source Toolbox for 3D Object Detection from Point Clouds},
	author={OpenPCDet Development Team},
	howpublished={\url{https://github.com/open-mmlab/OpenPCDet}},
	year={2020}
}
```