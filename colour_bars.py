import configparser
from sty import fg, bg, ef, rs
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image
import os
import colorcet as cc
# import misc
from rasterio.plot import show
import rasterio as raz
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Ini_dict:
    """ sorts out the ini settings"""

    def __init__(self):
        self.file_names = ''
        self.dir_path = ''
        self.single_file_1 = ''
        self.single_file_2 = ''
        self.section_io = 'io'
        self.section_params = 'params'
        self.section_lists = 'lists'
        self.section_bools = 'bools'
        self.io = ''
        self.params = ''
        self.lists = ''
        self.bools = ''
        self.an_ini_file = os.path.splitext(__file__)[0] + '.ini'

    def update_dicts(self):
        """
        update the dicts so they can be written to file
        """
        for section in [self.params, self.lists, self.bools]:
            for item, value in section.items():
                section[item] = getattr(self, item)

    def read(self):
        config = configparser.ConfigParser()
        # Note: case sensitive keys are required

        if os.path.isfile(self.an_ini_file):
            config.read(self.an_ini_file)
            self.dir_path = config.get(self.section_io, 'dir_path', fallback='')
            self.file_names = config.get(self.section_io, 'file_names', fallback='')
            self.single_file_1 = config.get(self.section_io, 'single_file_1', fallback='')
            self.single_file_2 = config.get(self.section_io, 'single_file_2', fallback='')
            # s_gui_values['_browse_'].split(',')
            self.io = dict(config[self.section_io])
            self.params = dict(config[self.section_params])
            self.lists = dict(config[self.section_lists])
            self.bools = dict(config[self.section_bools])
            for section in [self.params, self.lists, self.bools]:
                for k, v in section.items():
                    setattr(self, k, v)

        else:
            print('did not find the file ' + self.an_ini_file)
            print('using default values')

    def write(self):
        """
        Note: sort out writing a default workable ini if none exists; use generic defaults
        :return: nothing
        """
        self.update_dicts()
        if not os.path.isfile(self.an_ini_file):
            open(self.an_ini_file, 'w').close()

        config = configparser.ConfigParser()
        config[self.section_io] = {}
        io = config[self.section_io]
        # Note: case sensitive keys are required

        io['dir_path'] = self.dir_path
        io['file_names'] = str(self.file_names)
        io['single_file_1'] = self.single_file_1
        io['single_file_2'] = self.single_file_2
        config[self.section_params] = dict(self.params)
        config[self.section_lists] = dict(self.lists)
        config[self.section_bools] = dict(self.bools)

        with open(self.an_ini_file, 'w') as a_file:
            config.write(a_file)

    def print(self):
        names = [self.section_io, self.section_params, self.section_lists, self.section_bools]
        i = 0
        print(fg.yellow + 'ini file: ' + self.an_ini_file + fg.rs)
        for section in [self.io, self.params, self.lists, self.bools]:
            print(fg.red + ef.italic + '[' + names[i] + ']' + rs.all)
            for k, v in section.items():
                print(fg.li_blue + k + fg.green + ' = ' + fg.rs + v)
            i += 1


def f_stem(full_file_name):
    """ Returns stem or basename of file without the path on the front or the extension"""
    f_name = os.path.basename(full_file_name)
    return os.path.splitext(f_name)[0]


def read_grid(grid_name):
    with raz.open(grid_name) as a_raster:
        grid = a_raster.read(masked=True)
        a_profile = a_raster.profile
    grid_array = raz.plot.reshape_as_image(grid)
    # print(type(grid_array))
    grid_extent = raz.plot.plotting_extent(a_raster)
    return grid_array, grid_extent, a_profile


def write_world_file(f_name, a_profile):
    """
    see: https://en.wikipedia.org/wiki/World_file

    """
    affine = a_profile.get('transform')
    with open(f_name, 'w') as a_world_file:
        a_world_file.write('{0:.10f}'.format(affine.a) + '\n')
        a_world_file.write('{0:.3f}'.format(affine.d) + '\n')
        a_world_file.write('{0:.3f}'.format(affine.b) + '\n')
        a_world_file.write('{0:.10f}'.format(affine.e) + '\n')
        a_world_file.write('{0:.10f}'.format(affine.c) + '\n')
        a_world_file.write('{0:.10f}'.format(affine.f) + '\n')
    print(fg.yellow + ef.i + 'saved: ' + f_name + rs.ef)
    print(fg.cyan + 'world file parameters:' + fg.yellow)
    print('{0:.10f}'.format(affine.a))
    print('{0:.3f}'.format(affine.d))
    print('{0:.3f}'.format(affine.b))
    print('{0:.10f}'.format(affine.e))
    print('{0:.10f}'.format(affine.c))
    print('{0:.10f}'.format(affine.f) + rs.fg)


def read_saga_palette(f_name):
    """
    takes a palette exported from SAGA and turns it into a RGB list scaled 0 to 1
    """
    a_list = []
    with open(f_name, 'r') as saga_pal:
        for line in saga_pal.readlines():
            line = line.split()
            a_list.append(line)
    del a_list[0:2]
    np_array = np.array(a_list, dtype=int)
    np_array = np_array / 255
    a_list = np_array.tolist()
    return a_list


def list_to_mpl_cmap(a_list, gamma=1.0, cmap_name='mpl_cmap'):
    """
    takes a RGB list and converts it intp an mpl cmap
    """

    from matplotlib.colors import LinearSegmentedColormap
    cmap_np_array = np.array(a_list)
    cmap_np_array = cmap_np_array
    a_cmap = LinearSegmentedColormap.from_list(cmap_name, cmap_np_array.tolist(), gamma=gamma)
    return a_cmap


def save_as_image(an_image_file, an_array, a_cmap, a_vmin=0, a_vmax=100):
    print(an_array.shape)
    an_array = an_array.reshape(an_array.shape[0], an_array.shape[1])
    an_array[an_array == -99999.0] = np.nan
    print(fg.cyan + 'grid data min: {0:2f} grid data max: {1:2f}'.format(an_array.min(), an_array.max()))
    matplotlib.image.imsave(an_image_file, an_array,
                            cmap=a_cmap, vmax=a_vmax, vmin=a_vmin,
                            pil_kwargs={'optimize': True,
                                        'dpi': (100, 100)})
    print(fg.yellow + 'saved: ' + image_file + rs.all)


def plot_grid(an_array, a_cmap, an_extent, a_gamma=1.0, a_vmin=0, a_vmax=100):
    mpl.rcParams['legend.fontsize'] = 8
    mpl.rcParams['axes.titlesize'] = 11
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    mpl.rcParams['savefig.dpi'] = 200
    mpl.rcParams['mathtext.fontset'] = 'stix'
    ax_list = []
    fig = plt.figure(figsize=(8, 8),
                     num=a_cmap.name + ' [gamma={0:.2f} vmin={1:.2f} vmax={2:.2f}]'.format(a_gamma, a_vmin, a_vmax))
    axd = fig.subplot_mosaic([['ax01']], sharex=False, )
    for k in axd:
        ax_list.append(axd[k])
    for ax in ax_list:
        ax.set_aspect(aspect='equal')
        ax.set_box_aspect(1)
    if 'ax01' in axd:
        line_1 = axd['ax01'].imshow(an_array, cmap=a_cmap, extent=an_extent, vmin=a_vmin, vmax=a_vmax,
                                    )
        axd['ax01'].grid()

        divider = make_axes_locatable(axd['ax01'])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        clb = plt.colorbar(line_1, cax=cax)
        clb.ax.tick_params(labelsize=8, direction='in', )
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    ini = Ini_dict()
    ini.read()
    ini.print()
    a_grid_file = ini.file_names.split(',')[0].strip()
    saga_palette = ini.single_file_1
    pal_list = read_saga_palette(ini.single_file_1)
    pal_name = f_stem(ini.single_file_1)
    # pal_list = cc.diverging_bwr_55_98_c37
    # pal_name = 'diverging_bwr_55_98_c37'
    cmap = list_to_mpl_cmap(pal_list, float(ini.gamma), cmap_name=pal_name)
    array, extent, profile = read_grid(a_grid_file)
    a_vmax = float(ini.a_vmax)
    plot_grid(array, cmap, extent, float(ini.gamma), a_vmin=0, a_vmax=a_vmax)
    image_file = ini.dir_path + os.sep + f_stem(a_grid_file) + ini.ext_image_file
    save_as_image(image_file, array, cmap, a_vmin=0, a_vmax=73)
    world_file = os.path.splitext(image_file)[0] + ini.ext_world_file
    write_world_file(world_file, profile)
    ini.write()
    input('press return to close...')

