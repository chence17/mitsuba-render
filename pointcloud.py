'''
Author: chence17 antonio.chan.cc@outlook.com
Date: 2022-11-02 20:20:04
LastEditors: chence17 antonio.chan.cc@outlook.com
LastEditTime: 2022-11-02 21:03:21
FilePath: /AcLibPy/visualization3d/pointcloud.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import matplotlib.pyplot as plt
import os
import numpy as np
import plyfile
import tqdm
from typing import Any, Dict, List
import xml.etree.ElementTree as etree
import mitsuba as mi
import argparse

# Modified from https://github.com/OuyangJunyuan/PointCloudRenderer
class PointcloudVisualizer(object):
    def __init__(self, hparams: argparse.Namespace) -> None:
        super(PointcloudVisualizer, self).__init__()
        self.hparams = hparams
        # load point cloud
        assert os.path.exists(hparams.pcd_file), f'input path {hparams.pcd_file} does not exist.'
        self.pts = self.load_from_ply(hparams.pcd_file, hparams.sample_number, hparams.sample_method)
        # process pts
        self.pts = self.pts[:, [2, 0, 1]]
        self.pts[:, 0] *= -1
        self.pts[:, 1] *= -1
        self.pts[:, 2] += 0.0125 # 抬升一点距离
        # create dict for mitsuba scene description.
        self.scene_dict = self.create_environment(hparams.sample_count, hparams.width, hparams.height, hparams.camera_file)
        self.scene_dict.update(self.create_objects(hparams.pts_size, hparams.pts_color))

    def farthest_point_sampling(self, pts: np.array, sample_number: int) -> np.ndarray:
        """
        Modified from: https://zhuanlan.zhihu.com/p/400427336
        Input:
            pts: pointcloud data, [N, 3]
            sample_number: number of samples
        Return:
            centroids: sampled pointcloud index, [sample_number,]
        """
        # 获取形状
        N, C = pts.shape
        # 采样点矩阵(npoint,)
        centroids = np.zeros(sample_number, dtype=np.uint32)
        # 采样点到所有点距离(N,)
        distance = np.ones(N) * 1e10
        #计算重心坐标 及 距离重心最远的点, 将距离重心最远的点作为第一个点
        barycenter = np.mean(pts, 0)
        dist = np.sum((pts - barycenter) ** 2, -1)
        farthest = np.argmax(dist)
        # 遍历
        for i in tqdm.trange(sample_number):
            # 更新第i个最远点
            centroids[i] = farthest
            # 取出这个最远点的xyz坐标
            centroid = pts[farthest, :]
            # 计算点集中的所有点到这个最远点的欧式距离
            dist = np.sum((pts - centroid) ** 2, -1)
            mask = dist < distance
            # 更新distance，记录样本中每个点距离所有已出现的采样点的最小距离
            distance[mask] = dist[mask]
            # 返回最远点索引
            farthest = np.argmax(distance)
        return centroids

    def load_from_ply(self, file_path: str, sample_number: int=0, sample_method: str='fps') -> np.ndarray:
        """
        Load points from a ply file
        :param filepath: The path of the file to read from
        :return: An ndarray (Nx3) containing all read points
        """
        plydata = plyfile.PlyData.read(file_path)
        pts = plydata['vertex'].data
        pts_np = np.array([[x, y, z] for x,y,z in pts])
        print(f'loaded pointclod shape: {pts_np.shape}.')
        if sample_number > 0 and sample_number < pts_np.shape[0]:
            if sample_method == 'fps':
                print(f'using farthest point sampling to sample {sample_number} points.')
                pt_indices = self.farthest_point_sampling(pts_np, sample_number)
            elif sample_method == "random":
                print(f'using random sampling to sample {sample_number} points.')
                pt_indices = np.random.choice(pts.shape[0], sample_number, replace=False)
            else:
                raise NotImplementedError(f'sample method {sample_method} is not implemented.')
            pts_np = pts_np[pt_indices, :]
            print(f'pointclod shape after sampling: {pts_np.shape}.')
        elif sample_number > pts_np.shape[0]:
            print(f'sample number {sample_number} > total number {pts_np.shape[0]}, no sampling is performed, use all points.')
        else:
            print(f'No sampling is performed, use all points.')
        return pts_np

    def convert_camera_meshlab2mitsuba(self, file_path: str, film: Dict[str, Any], sampler: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Reads meshlab camera properties from an xml file. To obtain those, simply press ctrl/cmd+c in meshlab, and paste into a text editor.
        If you paste multiple cameras into one file, make sure to have a valid document structure with one root node, e.g.
        <!DOCTYPE ViewState>
        <project>
        <VCGCamera ... />
        <ViewSettings ... />
        </project>

        :param file_path: Path to the xml file to read
        :return: A list of Mitsuba transforms
        """
        assert file_path.endswith('.xml'), 'Invalid camera file.'
        root_node = etree.parse(file_path).getroot()
        meshlab_cameras = [elem.attrib for elem in root_node if elem.tag == 'VCGCamera']
        mitsuba_cameras = []
        for c in meshlab_cameras:
            # Meshlab camera matrix is transposed
            matrix = np.array([
                float(elem) for elem in c['RotationMatrix'].split(' ')[:16]
            ]).reshape(4, 4).transpose()
            translation = [-float(elem) for elem in c['TranslationVector'].split(' ')]
            for i in range(3):
                matrix[i, 3] = translation[i]
            transform = mi.Transform4f(matrix)
            transform = transform @ transform.scale(mi.Vector3f(-1, 1, -1))
            transform = transform.matrix.numpy()[0]
            position = transform[:3, 3]
            look = transform[:3, :3].dot(np.array([0, 0, 1]))
            up = transform[:3, :3].dot(np.array([0, 1, 0]))
            # transform
            position = position[[2, 0, 1]]
            position[0] *= -1
            position[1] *= -1
            position[2] += 0.0125 # see init()
            look = look[[2, 0, 1]]
            look[0] *= -1
            look[1] *= -1
            look[2] += 0.0125 # see init()
            up = up[[2, 0, 1]]
            up[0] *= -1
            up[1] *= -1
            up[2] += 0.0125 # see init()
            mitsuba_cameras.append({
                "type": "perspective",
                "near_clip": 0.1,  # Distance to the near clip planes
                "far_clip": 100.0,  # Distance to the far clip planes
                "to_world": mi.ScalarTransform4f.look_at(
                    origin=position*1.5, # scale to make object small
                    target=position+look,
                    up=up
                    ),
                "film": film,
                "sampler": sampler
            })
        return mitsuba_cameras

    def create_environment(self, sample_count: int=256, width: int=1920, height: int=1080, camera_file: str=None) -> Dict[str, Any]:
        """
        Create the environment for the scene
        :param sample_count: Specify the sample times, the higher the better, which result have less noise.
        :return: A dict containing the environment description
        """
        # https://mitsuba2.readthedocs.io/en/latest/generated/plugins.html#integrators
        integer = {
            "type": "path",
            "max_depth": -1,  # Specifies the longest path depth in the generated output image, -1 means infinit.
            # "samples_per_pass": 4,
        }
        # https://mitsuba2.readthedocs.io/en/latest/generated/plugins.html#emitters
        # note that area light should be attached to a geometry object, as mentioned in the above link.
        area_emitter = {"type": "constant"}
        # https://mitsuba2.readthedocs.io/en/latest/generated/plugins.html#films
        film = {
            "type": "hdrfilm",  # high dynamic render
            "width": width,  # width of the camera snesor in pixel
            "height": height,  # height of the camera snesor in pixel
            "rfilter": {"type": "gaussian"}  # reconstruction filter
        }
        # https://mitsuba2.readthedocs.io/en/latest/generated/plugins.html#samplers
        sampler = {
            "type": "independent",  # independent sampling with a uniform distribution.
            "sample_count": sample_count  # the higher the better, which result have less noise.
        }
        # https://mitsuba2.readthedocs.io/en/latest/generated/plugins.html#sensors
        if camera_file is not None:
            mitsuba_cameras = self.convert_camera_meshlab2mitsuba(camera_file, film, sampler)
            assert len(mitsuba_cameras)>0, f'No camera found in {camera_file}.'
            camera = mitsuba_cameras[0]
        else:
            camera = {
                "type": "perspective",
                "near_clip": 0.1,  # Distance to the near clip planes
                "far_clip": 100.0,  # Distance to the far clip planes
                "to_world": mi.ScalarTransform4f.look_at(
                    origin=[0, 2, 0],
                    target=[0, 0, 0],
                    up=[0, 0, 1]
                    ),
                "film": film,
                "sampler": sampler
            }
        # scene dict
        scene = {
            "type": "scene",
            "integer": integer,
            "sensor": camera,
            "emitter": area_emitter
        }
        return scene

    def normalize_uint8_color(self, uint8_color_list: List) -> np.ndarray:
        """
        Normalize the colors to be in the range [0, 1]
        :param colors: The colors to normalize
        :return: The normalized colors
        """
        return [i / 255. for i in uint8_color_list]

    def get_predefined_spectrum_color(self, name: str) -> List:
        """
        Get a predefined spectrum from a name string.
        Currently supports: light_blue, cornflower_blue, orange
        :param name: The spectrum name
        :return: A Mitsuba Spectrum Color
        """
        if name == 'light_blue':
            return self.normalize_uint8_color([160, 180, 200])
        elif name == 'cornflower_blue':
            return self.normalize_uint8_color([100, 149, 237])
        elif name == 'orange':
            return self.normalize_uint8_color([200, 160, 0])
        else:
            raise ValueError

    def create_spheres(self, pts: np.ndarray, sphere_color: str = 'orange', sphere_radius: float = 1.) -> List:
        """
        Create little spheres at the pointcloud's points' locations.
        :param pointcloud: 3D pointcloud, as Nx3 ndarray
        :param spectrum: The color spectrum to use with the spheres' diffuse bsdf
        :param sphere_radius: The radius to use for each sphere
        :return: A list of mitsuba shapes, to be added to a Scene
        """
        spheres = []
        for row in pts:
            sphere = mi.load_dict({
                'type': 'sphere',
                'center': [float(row[0]), float(row[1]), float(row[2])],
                'radius': sphere_radius,
                'bsdf': {
                    'type': 'diffuse',
                    'reflectance': {
                        'type': 'srgb',
                        'color': self.get_predefined_spectrum_color(sphere_color)
                    }
                },
            })
            spheres.append(sphere)
        return spheres

    def create_objects(self, pts_size: float = 0.01, pts_color: str='orange') -> Dict[str, Any]:
        ground = {  # to make shadow
            "type": "rectangle",
            "bsdf": {
                'type': 'diffuse',
                'reflectance': {
                    'type': 'srgb',
                    'color': [1., 1., 1.]
                }
            },
            "to_world": mi.ScalarTransform4f.translate([0, 0, -0.5]) @ mi.ScalarTransform4f.scale([10, 10, 1]),
        }
        # ground plane that shadow projected to.
        objs = {"ground_plane": ground, }
        spheres = self.create_spheres(self.pts, pts_color, pts_size)
        # add rest to scene dict
        for i, obj in enumerate(spheres):
            objs[f"shape_{i+1}"] = obj
        return objs

    def render(self, show=False) -> None:
        # load scene from dict
        scene = mi.load_dict(self.scene_dict)
        sensor = scene.sensors()[0]  # select the first camera to have a view.
        scene.integrator().render(scene, sensor)
        # save the render result.
        film = sensor.film()
        self.image = film.bitmap(raw=True).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True)
        if show:
            self.show()

    def show(self) -> None:
        if hasattr(self, 'image'):
            plt.imshow(self.image)
            plt.show()
        else:
            print('Please run render() to generate rendered image data.')

    def save(self, save_file: str) -> None:
        if hasattr(self, 'image'):
            self.image.write(save_file)
        else:
            print('Please run render() to generate rendered image data.')

def parse_args():
    parser = argparse.ArgumentParser(description='Renser Pointcloud')
    parser.add_argument('-pcd_file', help='path to pointcloud file, support .ply file', type=str, required=True)
    parser.add_argument('-mi_var', help='set mitsuba.variant when rendering, set cuda_ad_rgb for GPU parallels if exists, set llvm_ad_rgb for CPU parallels', type=str, required=True, choices=mi.variants())
    parser.add_argument('-save_dir', help='folder to save rendered image', type=str, required=True)
    parser.add_argument('-sample_number', default=0, type=int, help='down-sampling number when load pcd, <=0 means no down-sampling')
    parser.add_argument('-sample_method', default='fps', type=str, help='down-sampling method, choose from random and fps', choices=['random', 'fps'])
    parser.add_argument('-sample_count', default=256, type=int, help='specify the sample times when rendering, the higher the better')
    parser.add_argument('-width', default=1080, type=int, help='rendered image width, default 1080')
    parser.add_argument('-height', default=1920, type=int, help='rendered image height, default 1920')
    parser.add_argument('-camera_file', default=None, type=str, help='path to camera.xml file, default None to use default camera')
    parser.add_argument('-pts_size', default=0.01, type=float, help='point size of each point in pointcloud when rendering')
    parser.add_argument('-pts_color', default='orange', type=str, help='point color of each point in pointcloud when rendering', choices=['light_blue', 'cornflower_blue', 'orange'])
    args = parser.parse_args()
    return args

if __name__=="__main__":
    hparams = parse_args()
    print(hparams)
    mi.set_variant(hparams.mi_var)
    assert os.path.isfile(hparams.pcd_file) and hparams.pcd_file.endswith('.ply'), f'Invalid pcd_file {hparams.pcd_file}'
    assert os.path.isdir(hparams.save_dir), f'Invalid save_dir {hparams.save_dir}'
    save_file = os.path.join(hparams.save_dir, os.path.basename(hparams.pcd_file).replace('.ply', '.png'))
    visualizer = PointcloudVisualizer(hparams)
    visualizer.render(show=False)
    visualizer.save(save_file)
