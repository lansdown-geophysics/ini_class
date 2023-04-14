import configparser

from sty import fg, bg, ef, rs
import cmasher as cmr
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image
from matplotlib.colors import LightSource
import os
import colorcet as cc

# from PIL import Image
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


class GIS_images:
    PINK = 213

    def __init__(self, array=None, extent=None, profile=None, norm_type=None, cmap=None,
                 vmin=None, vmax=None, bounds=None, norm_labels=None, do_plot=True,
                 do_shade=False, azdeg=315, altdeg=45 ):
        self.array = array
        self.extent = extent
        self.profile = profile
        self.cmap = cmap
        self.norm_type = norm_type
        self.norm = None
        self.vmin = vmin
        self.vmax = vmax
        self.bounds = bounds
        self.norm_labels = norm_labels
        self.do_plot = do_plot
        self.do_shade = do_shade
        self.im_array = np.ndarray
        self.im_name = ''
        self.cbar_title = None
        self.num = ''
        self.azdeg = azdeg
        self.altdeg = altdeg
        self.plot()

    def __write_world_file(self):
        """
        see: https://en.wikipedia.org/wiki/World_file

        """
        world_name = os.path.splitext(self.im_name)[0] + '.wld'
        affine = self.profile.get('transform')
        with open(world_name, 'w') as a_world_file:
            a_world_file.write('{0:.10f}'.format(affine.a) + '\n')
            a_world_file.write('{0:.3f}'.format(affine.d) + '\n')
            a_world_file.write('{0:.3f}'.format(affine.b) + '\n')
            a_world_file.write('{0:.10f}'.format(affine.e) + '\n')
            a_world_file.write('{0:.10f}'.format(affine.c) + '\n')
            a_world_file.write('{0:.10f}'.format(affine.f) + '\n')
        print(fg.yellow + ef.i + 'world file saved: ' + world_name + rs.ef)
        print(fg.cyan + 'world file parameters:' + fg.yellow)
        print('\t{0:.10f}'.format(affine.a))
        print('\t{0:.3f}'.format(affine.d))
        print('\t{0:.3f}'.format(affine.b))
        print('\t{0:.10f}'.format(affine.e))
        print('\t{0:.10f}'.format(affine.c))
        print('\t{0:.10f}'.format(affine.f) + rs.fg)

    def __write_colour_bar(self):
        c_bar_f_name = os.path.splitext(self.im_name)[0]+'_cbar_' + os.path.splitext(self.im_name)[1]
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.subplots_adjust(left=0.55, right=0.6)

        ax.tick_params(labelsize=15)

        cb = plt.colorbar(mpl.cm.ScalarMappable(norm=self.norm, cmap=self.cmap),
                          cax=ax,
                          # extend='max',
                          ticks=bounds,
                          spacing='proportional',
                          orientation='vertical', )
        if self.norm_labels is not None:
            cb.ax.set_yticks(self.norm_labels)
            cb.ax.set_yticklabels(self.norm_labels, fontsize=6)
        cb.set_label(label=self.cbar_title, size=10, )
        # cb.set_label(label='Slope (\N{degree sign})', size=25, )
        plt.savefig(c_bar_f_name, dpi=300, bbox_inches='tight', transparent=True)
        print(fg.yellow + ef.i + 'colour bar saved: ' + c_bar_f_name + rs.ef)

    def __make_norm(self):

        if self.norm_type == 'boundary':
            self.norm = mpl.colors.BoundaryNorm(self.bounds, self.cmap.N, extend='max')
            self.norm_labels = None
        if self.norm_type == 'minmax':
            self.bounds = None
            self.norm = mpl.colors.Normalize(self.vmin, self.vmax)
        if self.norm_type == 'powernorm':
            self.norm = mpl.colors.PowerNorm(3.0, vmin=self.vmin, vmax=self.vmax, )
        if self.norm_type == 'twoslope':
            self.norm = mpl.colors.TwoSlopeNorm(-25, vmin=self.vmin, vmax=self.vmax, )

    def __make_shade(self, shade_array):
        if self.do_shade:
            ls = LightSource(azdeg=self.azdeg, altdeg=self.altdeg)
            return ls.shade(shade_array, self.cmap, norm=self.norm)
        else:
            self.im_array = self.cmap(self.norm(self.im_array))
            self.azdeg = None
            self.altdeg = None
            return shade_array

    def write_image(self, im_name, cbar_label=None):
        """
        writes image , colour bar, and world file(.wld)
        :param cbar_label: text for cbar
        :param im_name: image name with extension
        """
        self.im_name = im_name
        self.cbar_title = cbar_label
        self.im_array = self.__make_shade(self.im_array)

        matplotlib.image.imsave(self.im_name, self.im_array, cmap=self.cmap, metadata={'Comment': self.num})
        print(fg.yellow + ef.i + 'image file saved : ' + self.im_name + rs.ef)
        self.__write_colour_bar()
        self.__write_world_file()

    def plot(self):
        self.__make_norm()
        self.array = self.array.reshape(self.array.shape[0], self.array.shape[1])
        self.im_array = self.array.copy()
        self.array = self.__make_shade(self.array)

        self.num = ('norm: ' + self.norm_type + ' | pal: ' + self.cmap.name
                    + ' | N: ' + str(self.cmap.N) + ' vmin: ' + str(self.vmin) + ' vmax: ' + str(self.vmax)
                    + ' | shade: ' + str(self.do_shade)
                    + ' az: ' + str(self.azdeg) + ' alt: ' + str(self.altdeg)
                    + ' | angus@lansdown-geophysics.co.uk')
        print(fg(self.PINK) + self.num + fg.rs)
        ax_list = []
        fig = plt.figure(figsize=(8, 8), num=self.num)
        axd = fig.subplot_mosaic([['ax01']], sharex=False, )
        for k in axd:
            ax_list.append(axd[k])
        for ax in ax_list:
            ax.set_aspect(aspect='equal')
            ax.set_box_aspect(1)
        if 'ax01' in axd:

            # axd['ax01'].imshow(self.array, cmap=self.cmap, extent=self.extent, norm=self.norm,)
            axd['ax01'].imshow(self.array, cmap=self.cmap, extent=self.extent, norm=self.norm)
            print(fg(self.PINK) + 'array shape: ' + str(self.array.shape) + fg.rs)
            # self.im_array = self.cmap(self.norm(self.array))
            # self.im_array = self.im_array.reshape(self.im_array.shape[0], self.im_array.shape[1], self.im_array.shape[3])
            # print(fg(self.PINK) + 'image array shape: ' + str(self.im_array.shape) + fg.rs)
            axd['ax01'].grid()
            divider = make_axes_locatable(axd['ax01'])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            clb = plt.colorbar(mpl.cm.ScalarMappable(norm=self.norm, cmap=self.cmap),
                               cax=cax,
                               ticks=self.bounds,
                               spacing='proportional',
                               orientation='vertical', )
            clb.ax.tick_params(labelsize=8, direction='in')

            if self.norm_labels is not None:
                clb.ax.set_yticks(self.norm_labels)
                clb.ax.set_yticklabels(self.norm_labels)
            if self.do_plot:
                plt.show()
            # return im_array, norm


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


def read_saga_palette(f_name: str) -> list:
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


def list_to_mpl_cmap(a_list: list, gamma: float = 1.0, cmap_name: str = 'mpl_cmap') -> object:
    """
    takes a RGB list and converts it into a mpl cmap
    """
    from matplotlib.colors import LinearSegmentedColormap
    a_cmap = LinearSegmentedColormap.from_list(cmap_name, a_list, gamma=gamma)
    return a_cmap


def resample_colour_map(a_cmap, n_cols: int) -> object:

    from matplotlib.colors import ListedColormap
    new = ListedColormap(a_cmap(np.linspace(0.0, 1.0, n_cols)))
    new.name = a_cmap.name
    return new


def string_to_bool(string: str) -> bool:
    try:
        match(string.lower()):
            case 'true':
                return True
            case 'false':
                return False
            case _:
                raise ValueError()
    except ValueError:
        print(fg(196) + ef.i + ' value in the ini file not recognised as boolean: '
              + fg(213) + ' "' + string + '" ' + rs.all)


def bool_to_string(a_bool: bool) -> str:
    try:
        match a_bool:
            case True:
                return 'True'
            case False:
                return 'False'
            case _:
                raise ValueError()
    except ValueError:
        print(fg(196) + ef.i + ' value passed not recognised: ' + rs.all)


if __name__ == '__main__':
    image_ext = '.png'

    ini = Ini_dict()
    ini.read()
    ini.print()
    # Note: set variables from ini file parameters

    try:

        saga_palette = ini.single_file_1
        gamma_func = float(ini.gamma)
        min_val = float(ini.a_vmin)
        max_val = float(ini.a_vmax)
        if min_val == max_val:
            min_val = None
            max_val = None
        n_colours = int(ini.n_colours)
        norm_labels = ini.norm_labels.split(',')
        norm_labels = [float(x) for x in norm_labels]
        norm_labels = None
        normalisation_type = ini.norm_type
        screen_plot = string_to_bool(ini.do_plot)
        shade_image = string_to_bool(ini.do_shade)
        use_inbuilt_palette = string_to_bool(ini.do_use_inbuilt_palette)
        sun_az = int(ini.sun_az)
        sun_alt = int(ini.sun_alt)
        print(shade_image)
    except AttributeError as e:
        print(fg(196) + ef.i + ' missing parameter in the ini file: ' + str(e).strip() + rs.all)
        exit()
    if use_inbuilt_palette:
        # Note: https://cmasher.readthedocs.io/user/cmap_overviews/cmr_cmaps.html
        palette_list = cmr.neon_r.colors
        palette_name = cmr.neon_r.name
        # palette_list = cc.linear_gow_60_85_c27
        # palette_name = 'linear_gow_60_85_c27'
    else:
        palette_list = read_saga_palette(ini.single_file_1)
        palette_name = f_stem(ini.single_file_1)
    print(fg.yellow + '\nusing : "' + ef.i + palette_name + '" palette is inbuilt=' + bool_to_string(use_inbuilt_palette) + rs.all)
    colour_map = list_to_mpl_cmap(palette_list, gamma_func, cmap_name=palette_name)

    if normalisation_type == 'boundary':
        bounds = ini.bounds.split(',')
        bounds = [float(x) for x in bounds]
        n_colours = len(bounds) + 1
    else:
        bounds = None
    cmap = resample_colour_map(colour_map, n_colours)
    # a_grid_file = ini.file_names.split(',')[0].strip()
    for a_grid_file in ini.file_names.split(','):
        print(fg.yellow + ef.i + '\nreading : ' + a_grid_file + '\n info follows:' + rs.all)
        # print(a_grid_file)
        array, extent, profile = read_grid(a_grid_file)
        print(fg(213) + 'gdal driver: ' + profile.get('driver') + '    crs: ' + str(profile.get('crs')) + fg.rs)

        image_file = ini.dir_path + os.sep + f_stem(a_grid_file) + image_ext
        gis = GIS_images(array=array, extent=extent, profile=profile, norm_type=normalisation_type,
                         cmap=cmap, vmin=min_val, vmax=max_val, bounds=bounds, norm_labels=norm_labels,
                         do_plot=screen_plot, do_shade=shade_image, azdeg=sun_az, altdeg=sun_alt)
        gis.write_image(image_file, 'Depth (m)')

    input('press return to close...')
