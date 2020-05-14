import numpy as np
import fresnel
import freud
import io
import PIL
import sys
import IPython
import math


device = fresnel.Device(mode='cpu');
preview_tracer = fresnel.tracer.Preview(device, 300, 300)
path_tracer = fresnel.tracer.Path(device, 300, 300)

blue = fresnel.color.linear([0.25,0.5,1])*0.9;
orange = fresnel.color.linear([1.0,0.714,0.169])*0.9


class Cassandra_frame():
    def __init__(self, types, positions, box):
        self.types = types
        self.positions = positions
        self.box = box


def make_traj(xyzfile, boxfile):
    d_xyz = np.dtype([("atom", np.unicode_, 8), ("xyz", "d", 3)])

    with open(xyzfile, "r") as f:
        xyzlines = f.readlines()

    with open(boxfile, "r") as f:
        boxlines = f.readlines()

    i = 0
    boxes = []
    while True:
        try:
            line = boxlines[i]
        except IndexError:
            break
        if len(line.split()) == 3:
            boxmat = np.array(
                [line.split(),boxlines[i+1].split(), boxlines[i+2].split()],
                dtype="float64"
            )
            boxes.append(freud.box.Box.from_matrix(boxmat))
            i += 3
        i += 1

    traj = []
    i = 0
    step = 0
    while True:
        try:
            n_atoms = int(xyzlines[i])
        except IndexError:
            break
        #print(xyzlines[i+1]) # this should print the step
        arr = xyzlines[i+2:i+2+n_atoms]
        data = np.genfromtxt(arr, dtype = d_xyz)
        atoms = np.array([i["atom"] for i in data], dtype="U8")
        xyz = np.stack([i["xyz"] for i in data])
        frame = Cassandra_frame(atoms, xyz, boxes[step])
        traj.append(frame)
        i += 2+n_atoms
        step += 1
    return traj


def render_sphere_frame(frame, height=None):
    """
    Modified from hoomd-examples/ex_render.py to work with the
    Cassandra_frame class
    """

    if height is None:
        if hasattr(frame, 'configuration'):
            Ly = frame.configuration.box[1]
            height = Ly * math.sqrt(3)
        else:
            Ly = frame.box.Ly;
            height = Ly * math.sqrt(3)

    scene = fresnel.Scene(device)
    scene.lights = fresnel.light.cloudy();
    g = fresnel.geometry.Sphere(scene, position=frame.positions, radius=np.ones(len(frame.types))*0.5)
    g.material = fresnel.material.Material(solid=0.0, color=blue, primitive_color_mix=1.0, specular=1.0, roughness=0.2)
    g.outline_width = 0.07
    scene.camera = fresnel.camera.orthographic(position=(height, height, height), look_at=(0,0,0), up=(0,1,0), height=height)

    g.color[frame.types == "C"] = blue;
    g.color[frame.types == "_C"] = blue;
    g.color[frame.types == "H"] = orange;

    scene.background_color = (1,1,1)

    return path_tracer.sample(scene, samples=64, light_samples=20)


def display_movie(frame_gen, traj, gif = None):
    a = frame_gen(traj[0], height = 30);

    if tuple(map(int, (PIL.__version__.split(".")))) < (3,4,0):
        print("Warning! Movie display output requires pillow 3.4.0 or newer.")
        print("Older versions of pillow may only display the first frame.")

    im0 = PIL.Image.fromarray(a[:,:, 0:3], mode='RGB').convert("P", palette=PIL.Image.ADAPTIVE);
    ims = [];
    for f in traj[1:]:
        a = frame_gen(f, height = 30);
        im = PIL.Image.fromarray(a[:,:, 0:3], mode='RGB')
        im_p = im.quantize(palette=im0);
        ims.append(im_p)

    if gif:
        im0.save(gif, 'gif', save_all=True, append_images=ims, duration=1000, loop=0)
        return
    if (sys.version_info[0] >= 3):
        size = len(io.BytesIO().getbuffer())/1024;
        if (size > 2000):
            print("Size:", size, "KiB")

    return IPython.display.display(IPython.display.Image(data=io.BytesIO().getvalue()))
