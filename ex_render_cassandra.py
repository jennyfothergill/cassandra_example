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

cpk_colors = {
    "H": fresnel.color.linear([1.00, 1.00, 1.00]),  # white
    "C": fresnel.color.linear([0.30, 0.30, 0.30]),  # grey
    "N": fresnel.color.linear([0.13, 0.20, 1.00]),  # dark blue
    "O": fresnel.color.linear([1.00, 0.13, 0.00]),  # red
    "F": fresnel.color.linear([0.12, 0.94, 0.12]),  # green
    "Cl": fresnel.color.linear([0.12, 0.94, 0.12]),  # green
    "Br": fresnel.color.linear([0.60, 0.13, 0.00]),  # dark red
    "I": fresnel.color.linear([0.40, 0.00, 0.73]),  # dark violet
    "He": fresnel.color.linear([0.00, 1.00, 1.00]),  # cyan
    "Ne": fresnel.color.linear([0.00, 1.00, 1.00]),  # cyan
    "Ar": fresnel.color.linear([0.00, 1.00, 1.00]),  # cyan
    "Xe": fresnel.color.linear([0.00, 1.00, 1.00]),  # cyan
    "Kr": fresnel.color.linear([0.00, 1.00, 1.00]),  # cyan
    "P": fresnel.color.linear([1.00, 0.60, 0.00]),  # orange
    "S": fresnel.color.linear([1.00, 0.90, 0.13]),  # yellow
    "B": fresnel.color.linear([1.00, 0.67, 0.47]),  # peach
    "Li": fresnel.color.linear([0.47, 0.00, 1.00]),  # violet
    "Na": fresnel.color.linear([0.47, 0.00, 1.00]),  # violet
    "K": fresnel.color.linear([0.47, 0.00, 1.00]),  # violet
    "Rb": fresnel.color.linear([0.47, 0.00, 1.00]),  # violet
    "Cs": fresnel.color.linear([0.47, 0.00, 1.00]),  # violet
    "Fr": fresnel.color.linear([0.47, 0.00, 1.00]),  # violet
    "Be": fresnel.color.linear([0.00, 0.47, 0.00]),  # dark green
    "Mg": fresnel.color.linear([0.00, 0.47, 0.00]),  # dark green
    "Ca": fresnel.color.linear([0.00, 0.47, 0.00]),  # dark green
    "Sr": fresnel.color.linear([0.00, 0.47, 0.00]),  # dark green
    "Ba": fresnel.color.linear([0.00, 0.47, 0.00]),  # dark green
    "Ra": fresnel.color.linear([0.00, 0.47, 0.00]),  # dark green
    "Ti": fresnel.color.linear([0.60, 0.60, 0.60]),  # grey
    "Fe": fresnel.color.linear([0.87, 0.47, 0.00]),  # dark orange
    "default": fresnel.color.linear([0.87, 0.47, 1.00]),  # pink
}

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
    g.material = fresnel.material.Material(solid=0.0, primitive_color_mix=1.0, specular=1.0, roughness=0.2)
    g.outline_width = 0.07
    scene.camera = fresnel.camera.orthographic(position=(height, height, height), look_at=(0,0,0), up=(0,1,0), height=height)

    for typename in set(frame.types):
        try:
            g.color[frame.types == typename] = cpk_colors[
                    typename[:2].strip("_")
                    ];
        except KeyError:
            g.color[frame.types == typename] = cpk_colors[
                    "default"
                    ];


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
