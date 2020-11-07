"""
(c) 2020 Spencer Rose, MIT Licence
MLP Landscape Classification Tool (MLP-LCT)
An evaluation of deep learning semantic segmentation for
land cover classification of oblique ground-based photography
MSc. Thesis 2020.
<http://hdl.handle.net/1828/12156>
Spencer Rose <spencerrose@uvic.ca>, June 2020
University of Victoria

File: params.py
    Application parameters.

Notes: Extra parameters loaded from 'settings.json' in the
root directory.
"""




''' Parse model configuration '''
config, unparsed, parser = get_config()
