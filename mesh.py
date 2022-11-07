'''
Author: chence17 antonio.chan.cc@outlook.com
Date: 2022-11-02 20:57:32
LastEditors: chence17 antonio.chan.cc@outlook.com
LastEditTime: 2022-11-04 15:21:22
FilePath: /AcLibPy/visualization3d/mesh.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import matplotlib.pyplot as plt
import os
import numpy as np
import plyfile
from typing import Any, Dict, List
import xml.etree.ElementTree as etree
import mitsuba as mi
import pymeshlab
import argparse

# Modified from https://github.com/OuyangJunyuan/PointCloudRenderer
class MeshVisualizer(object):
    def __init__(self, hparams: argparse.Namespace) -> None:
        super(MeshVisualizer, self).__init__()
        self.hparams = hparams
        # load mesh
        assert os.path.exists(hparams.mesh_file), f'input path {hparams.mesh_file} does not exist.'
        self.mesh_file = self.convert_mesh(hparams.mesh_file)
        # create dict for mitsuba scene description.
        self.scene_dict = self.create_environment(hparams.sample_count, hparams.width, hparams.height, hparams.camera_file)
        self.scene_dict.update(self.create_objects(self.mesh_file, hparams.mesh_color))

    def convert_mesh(self, mesh_file: str) -> str:
        if mesh_file.endswith('.obj'):
            mesh_file_ply = mesh_file.replace('.obj', '.ply')
            if os.path.exists(mesh_file_ply):
                print(f'{mesh_file_ply} exists, no converting.')
            else:
                ms = pymeshlab.MeshSet()
                ms.load_new_mesh(mesh_file)
                ms.save_current_mesh(mesh_file.replace('.obj', '.ply'))
                print(f'{mesh_file} is converted to {mesh_file_ply}.')
        else:
            mesh_file_ply = mesh_file
            print('ply file, no converting.')
        save_file = mesh_file_ply.replace('.ply', '_mitsuba.ply')
        if os.path.exists(save_file):
            print(f'{save_file} exists, no converting.')
        else:
            print(f'{save_file} does not exist, converting...')
            plydata = plyfile.PlyData.read(mesh_file_ply)
            # process pts
            pts = plydata['vertex'].data
            pts_np = np.array([[x, y, z] for x,y,z in pts])
            pts_np = pts_np[:, [2, 0, 1]]
            pts_np[:, 0] *= -1
            pts_np[:, 1] *= -1
            pts_np[:, 2] += 0.0125 # 抬升一点距离
            points = [(pts_np[i,0], pts_np[i,1], pts_np[i,2]) for i in range(pts_np.shape[0])]
            vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
            vertex = plyfile.PlyElement.describe(vertex, 'vertex', comments=['vertices'])
            face = plydata['face']
            plyfile.PlyData([vertex, face]).write(mesh_file_ply.replace('.ply', '_mitsuba.ply'))
            print(f'{save_file} saved.')
        return save_file

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

    def create_objects(self, mesh_file: str, mesh_color: str = 'orange') -> Dict[str, Any]:
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
        # objs
        objs = {
            "ground_plane": ground, # ground plane that shadow projected to.
            "instance": { # mesh
                'type': 'ply',
                'filename': mesh_file,
                'face_normals': True,
                'bsdf': {
                    'type': 'twosided',
                    'material': {
                        'type': 'diffuse',
                        'reflectance': {
                            'type': 'srgb',
                            'color': self.get_predefined_spectrum_color(mesh_color)
                            }
                        }
                    }
                }
            }
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
    parser = argparse.ArgumentParser(description='Renser Mesh')
    parser.add_argument('-mesh_file', help='path to mesh file, support .ply, .obj file', type=str, required=True)
    parser.add_argument('-mi_var', help='set mitsuba.variant when rendering, set cuda_ad_rgb for GPU parallels if exists, set llvm_ad_rgb for CPU parallels', type=str, required=True, choices=mi.variants())
    parser.add_argument('-save_dir', help='folder to save rendered image', type=str, required=True)
    parser.add_argument('-sample_count', default=256, type=int, help='specify the sample times when rendering, the higher the better')
    parser.add_argument('-width', default=1080, type=int, help='rendered image width, default 1080')
    parser.add_argument('-height', default=1920, type=int, help='rendered image height, default 1920')
    parser.add_argument('-camera_file', default=None, type=str, help='path to camera.xml file, default None to use default camera')
    parser.add_argument('-mesh_color', default='orange', type=str, help='color of mesh when rendering', choices=['light_blue', 'cornflower_blue', 'orange'])
    args = parser.parse_args()
    return args

if __name__=="__main__":
    hparams = parse_args()
    print(hparams)
    mi.set_variant(hparams.mi_var)
    assert os.path.isfile(hparams.mesh_file) and (hparams.mesh_file.endswith('.ply') or hparams.mesh_file.endswith('.obj')), f'Invalid pcd_file {hparams.mesh_file}'
    assert os.path.isdir(hparams.save_dir), f'Invalid save_dir {hparams.save_dir}'
    if hparams.mesh_file.endswith('.ply'):
        save_file = os.path.join(hparams.save_dir, os.path.basename(hparams.mesh_file).replace('.ply', '.png'))
    elif hparams.mesh_file.endswith('.obj'):
        save_file = os.path.join(hparams.save_dir, os.path.basename(hparams.mesh_file).replace('.obj', '.png'))
    else:
        raise NotImplementedError(f"Unsupported mesh file: {hparams.mesh_file}.")
    visualizer = MeshVisualizer(hparams)
    visualizer.render(show=False)
    visualizer.save(save_file)
