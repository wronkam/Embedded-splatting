import ipywidgets as widgets

def get_default():
    return "train.py -s /content/Embedded-splatting/nerf_synthetic/drums --notebook --test-every-n 500 --iterations 15000 -r 8 --ff-args type positional layers 2 width 128 input_size 4 embeding_size 16 learnable True init 200 rand_color True  residual -1"
def get_widgets():
    items = {}
    layout = widgets.Layout(display='flex',
                            width='80%')
    items['input_size']=widgets.IntSlider(value=3, min=1, max=32, step=1, description='Input size:',layout=layout)
    items['embedding_size'] = widgets.FloatLogSlider(base=2, value=3, min=0, max=9, step=1, description='Embedding size:',readout_format='6d',style={'description_width':'initial'},layout=layout)
    def show(val1,val2):
        print(f'Resultant network input size: {int(val1*val2)}')
    items['result_size'] = widgets.interactive_output(show, {
        'val1':items['input_size'],'val2': items['embedding_size']})
    items['width'] = widgets.FloatLogSlider(base=2, value=256, min=3, max=12, step=1, description='Width:',readout_format='6d',layout=layout)
    items['layers'] = widgets.IntSlider(value=2, min=1, max=16, step=1, description='Layers:',layout=layout)
    items['iterations'] = widgets.IntSlider(value=30000, min=2500, max=120000, step=2500, description='Iterations:',layout=layout)
    items['dataset'] = widgets.Dropdown(options=['hotdog', 'materials', 'ficuse', 'lego', 'mic','drums', 'chair', 'ship'],value='drums',description='Dataset:',layout=layout)
    items['sampling'] = widgets.IntText(value=500,description='Sample every N iterations:',disabled=False,style={'description_width':'initial'},layout=layout)
    items['embedding_type'] = widgets.ToggleButtons(options=['none', 'net', 'fft', 'gff', 'positional'],description='Embedding type:',value='positional',
                                                    tooltips=['No embedding or network', 'Simple network with no embedding', 'Fast fourier transform','Gaussian fourier feature','Positional embedding'],layout=layout)
    items['learnable'] = widgets.Checkbox(value=True,description='Learnable embedding',disabled=False,indent=False,layout=layout)
    items['init'] = widgets.IntSlider(value=300, min=0, max=2500, step=100, description='Iterations of initialization matching:',style={'description_width':'initial'},layout=layout)
    items['rand_color'] = widgets.Checkbox(value=True,description='Random initial color',disabled=False,indent=False,layout=layout)
    items['normalize'] = widgets.Checkbox(value=False,description='Normalize embedding',disabled=False,indent=False,layout=layout)
    items['resolution'] = widgets.Dropdown(options=[('Full',1),('1/2',2),
                                                    ('1/4',4),('1/8',8),('1/16',16)],value=1,description='Resolution of images:',style={'description_width':'initial'},layout=layout)
    items['residual'] = widgets.Dropdown(options=[('None (recommended)',-1),('TanH(Y)*S+T',0),
                                                  ('[TanH(Y)+X]*S+T',1),('[TanH(Y)*S+T]+X',2),],value=-1,description='Residual mode:',style={'description_width':'initial'},layout=layout)

    return items


def show_widgets(items, command):
    import ipywidgets as widgets
    pass
    button = widgets.Button(
        description='Save config',
        disabled=False,
        button_style='',
        tooltip='Saves chosen options to the command',
        icon='check'
    )
    output = widgets.Output()
    def save_config(b,command=command,output=output):
        with output:
            command['text'] = "train.py -s /content/Embedded-splatting/nerf_synthetic/{} --notebook --test-every-n {} --iterations {} -r {} {}".format(
                items['dataset'].value,
                int(items['sampling'].value),
                int(items['iterations'].value),
                int(items['resolution'].value),
                "" if items['embedding_type'].value=="none" else "--ff-args {} {} {} {} {} {} {} {} {} {}".format(
                    f"type {items['embedding_type'].value}",
                    f"layers {int(items['layers'].value)}",
                    f"width {int(items['width'].value)}",
                    f"input_size {int(items['input_size'].value)}",
                    f"embeding_size {int(items['embedding_size'].value)}",
                    f"{'learnable True' if items['learnable'].value else ''}",
                    f"init {int(items['init'].value)}",
                    f"{'rand_color True' if items['rand_color'].value else ''}",
                    f"{'normalize True' if items['normalize'].value else ''}",
                    f"residual {int(items['residual'].value)}",
                )
            )
            print("Command:",command['text'])
        return command
    out = widgets.VBox([v for k,v in items.items()])
    button.on_click(save_config)

    return out,button,output
