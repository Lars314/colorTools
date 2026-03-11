import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd

def g(w, mu, t1, t2):
    if w < mu:
        return np.exp(-1*(t1**2)*((w-mu)**2)/2)
    else:
        return np.exp(-1*(t2**2)*((w-mu)**2)/2)

def a_cmf_x(w):
    return 1.0*(1.056 * g(w, 599.8, 0.0264, 0.0323) \
           + 0.362 * g(w, 442.0, 0.0624, 0.0374) \
           - 0.065 * g(w, 501.1, 0.049, 0.0382) )

def a_cmf_y(w):
    return 1.0*(0.821 * g(w, 568.8, 0.0213, 0.0247) \
           + 0.286 * g(w, 530.9, 0.0613, 0.0322))
def a_cmf_z(w):
    return 1.0*(1.217 * g(w, 437.0, 0.0845, 0.0278) \
           + 0.681 * g(w, 459.0, 0.0385, 0.0725))

def colorFader_hex(c1,c2,mix=0):
    """
    Thanks to Markus Dutschke on StackOverflow for this nice solution
    https://stackoverflow.com/questions/25668828/how-to-create-colour-gradient-in-python
    """
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

def colorFader_rgb(c1,c2,mix=0):
    """
    Thanks to Markus Dutschke on StackOverflow for this nice solution
    https://stackoverflow.com/questions/25668828/how-to-create-colour-gradient-in-python
    """
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_rgb((1-mix)*c1 + mix*c2)

def wavelength_to_xyz(w):
    #if w < 380 or w > 700:
    #    return [0.0, 0.0, 0.0]
    
    X = a_cmf_x(w)
    Y = a_cmf_y(w)
    Z = a_cmf_z(w)
    """if X < 0.002:
        x = 0
    else:
        x = X
    if Y < 0.002:
        y = 0
    else:
        y = Y
    if Z < 0.002:
        z = 0
    else:
        z = Z"""
    return [X, Y, Z]

def wavelength_to_normalized_xyz(w):
    x = a_cmf_x(w) / 1.0559468177057083
    y = a_cmf_y(w) / 0.9980889739887843
    z = a_cmf_z(w) / 1.7840200399158705
    
    return [x, y, z]

def xyz_from_xy(x, y):
    """Return the vector (x, y, 1-x-y)."""
    return np.array((x, y, 1-x-y))

def maximize_contrast(hex_color: str) -> str:
    """
    Written by chat GPT.
    Returns a grayscale color (#000000 or #FFFFFF) that maximizes contrast 
    with the given hexadecimal background color.
    
    Args:
        hex_color (str): A color in hexadecimal format (e.g., "#RRGGBB").
    
    Returns:
        str: A grayscale color ("#000000" or "#FFFFFF") for optimal contrast.
    """
    # Remove the hash symbol if present
    hex_color = hex_color.lstrip("#")
    
    # Convert hex to RGB components
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    
    # Calculate the perceived luminance using the standard formula
    # Y = 0.2126*R + 0.7152*G + 0.0722*B (normalized to 0-255 range)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    
    # Choose black or white based on the luminance
    return "#FFFFFF" if luminance < 128 else "#000000"


class ColorSystem:
    """
    A color system defined by the CIE x, y, and z=1-x-y coordinates of its
    three primary illumants and its "white point".

    This class was originally written by Christian of SciPython, on 27 Mar 2016:
    https://scipython.com/blog/converting-a-spectrum-to-a-colour/

    It is modified by Lars Borchert in order to use an analytical approximation
    to the color matching function, as described on the CEI 1931 color space
    Wikipedia page: https://en.wikipedia.org/wiki/CIE_1931_color_space
    """    

    def __init__(self, red, green, blue, white):
        """Initialise the ColourSystem object.

        Pass vectors (ie NumPy arrays of shape (3,)) for each of the
        red, green, blue  chromaticities and the white illuminant
        defining the colour system.
        """

        # Chromaticities
        self.red = red
        self.green = green
        self.blue = blue
        self.white = white
        # The chromaticity matrix (rgb -> xyz) and its inverse
        self.M = np.vstack((self.red, self.green, self.blue)).T 
        self.MI = np.linalg.inv(self.M)
        # White scaling array
        self.wscale = self.MI.dot(self.white)
        # xyz -> rgb transformation matrix
        self.T = self.MI / self.wscale[:, np.newaxis]

    def xyz_to_rgb(self, xyz, out_fmt=None):
        """Transform from xyz to rgb representation of colour.

        The output rgb components are normalized on their maximum
        value. If xyz is out the rgb gamut, it is desaturated until it
        comes into gamut.

        By default, fractional rgb components are returned; if
        out_fmt='html', the HTML hex string '#rrggbb' is returned.

        """

        rgb = self.T.dot(xyz)
        if np.any(rgb < 0):
            # We're not in the RGB gamut: approximate by desaturating
            w = - np.min(rgb)
            rgb += w
        """do_normalize = False
        for i in range(0, len(rgb)):
            if rgb[i] < 0:
                rgb[i] = 0
            elif rgb[i] > 1:
                do_normalize = True"""
        
        #if not np.all(rgb==0):
        #if do_normalize:
            # Normalize the rgb vector
        rgb /= np.max(rgb)

        if out_fmt is None:
            return rgb
        elif out_fmt.lower() == 'html' or out_fmt.lower() == 'hex':
            return self.rgb_to_hex(rgb)
        return rgb

    def rgb_to_hex(self, rgb):
        """Convert from fractional rgb values to HTML-style hex string."""

        hex_rgb = (255 * rgb).astype(int)
        return '#{:02x}{:02x}{:02x}'.format(*hex_rgb)

    def wavelength_to_rgb(self, w, out_fmt=None):
        """Convert a wavelength w in nanometers to an rgb value"""
        xyz = wavelength_to_xyz(w)
        #print(xyz)
        rgb = self.xyz_to_rgb(xyz, out_fmt)
        return rgb


class ColorPalette:
    """
    """
    def __init__(self, n_colors, cmap=None, c1=None, c2=None, verbose=True):
        """
        """
        # setup the basic data
        self.cmap = cmap
        self.c1 = c1
        self.c2 = c2
        self.n_colors = n_colors
        self.df = None

        # initialize the colors and make the color dataframe
        # if we are using a cmap, make the colors from that
        if cmap is not None:
            self._generate_colors_cmap()
        elif (c1 is not None) and (c2 is not None):
            self._generate_colors_between()
        else:
            print("Error! You must provide either a cmap or start and end " +\
                  "colors in order to generate a palette.")

        # print the colors right away if people want that (yes by default)
        if verbose:
            self.print_color_text()
            self.print_color_figure()

    def _format_colors(self):
        """
        """
        # format them into a nice dataframe
        cdicts = []
        i = 1
        for color in self.colors:
            # color index in this space
            this_dict = {"i":i}
            # define the color in hexadecimal and RGB format
            this_dict["hex"] = mpl.colors.rgb2hex(color)
            this_dict["R"] = color[0]*255
            this_dict["G"] = color[1]*255
            this_dict["B"] = color[2]*255
            # save this dictionary
            cdicts.append(this_dict)
            # iterate
            i += 1
        # pandas dataframe to hold the color information
        self.df = pd.DataFrame(cdicts)

    def _generate_colors_between(self):
        """
        """
        self.colors = []
        for i in range(self.n_colors):
            this_color = colorFader_rgb(self.c1, self.c2, i/(self.n_colors-1))
            self.colors.append(this_color)
        self._format_colors()
        
    def _generate_colors_cmap(self):
        """
        """
        # get a list of the colors in the palette
        self.colors = self.cmap(np.linspace(0, 1, self.n_colors))
        self._format_colors()

    def print_color_text(self, do_print=True):
        """
        """
        text = ""
        for item in self.df.iterrows():
            row = item[1]
            i_str = "i = {:>2}".format(row["i"])
            hex_str = "Hex = " + row["hex"]
            rgb_str = "RGB = ({0:>3.0f}, {1:>3.0f}, {2:>3.0f})".format(
                row["R"], row["G"], row["B"])
            row = i_str + "    " + hex_str + "    " + rgb_str + "\n"
            text += row
        if do_print:
            print(text)
        return text
        

    def print_color_figure(self, figwidth=10, return_fig_and_ax=False, fig=None,
                           ax=None):
        """
        """
        x = np.linspace(1, self.n_colors, self.n_colors)
        if fig == None:
            fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(figwidth, 3)
        ax.set_xlim(0.5, self.n_colors+0.5)
        ax.set_ylim(0, 1)
        
        for item in self.df.iterrows():
            row = item[1]
            ax.text(x=row["i"], y=0.1, s=row["i"], horizontalalignment="center",
                    color=maximize_contrast(row["hex"]))

        ax.bar(x=x, height=[1]*self.n_colors, color=self.colors, width=1)
    
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])

        fig.tight_layout()

        if return_fig_and_ax:
            return fig, ax
