import os
import glob
import base64



def print_img(image_path, output_file):
    """
    Encodes an image into html.
    image_path (str): Path to image file
    output_file (file): Output html page
    """
    if os.path.exists(image_path):
        img = base64.b64encode(open(image_path, "rb").read()).decode("utf-8")
        print('<img src="data:image/png;base64,{0}">'.format(img), file=output_file)




# save_dir = './results/2D_green'
save_dir = '/home/cc/students/csguests/chendudai/Thesis/repos/clipseg/results/'
html_out = open(os.path.join(save_dir, "results.html"), "w")
print('<head><meta charset="UTF-8"></head>', file=html_out)
print("<h1>Results</h1>", file=html_out)
names = ['_window', '_statue', '_door', '_facade', '_tower', '_top_part', '_spire']

folders = os.listdir(save_dir)
for folder in folders:
    image_paths = glob.glob(os.path.join(save_dir, folder, "*.png"))
    for name in names:
        for image_path in image_paths:
            if name in image_path:
                print(f"<br><b>{os.path.basename(image_path)}</b><br>", file=html_out)
                print_img(image_path, html_out)

print("<hr>", file=html_out)
html_out.close()