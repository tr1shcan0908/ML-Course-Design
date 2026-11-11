import numpy as np
import pandas as pd
from PIL import Image
import torchvision
import os
"""
In order to reduce the submission file size, our metric uses run-length encoding on the pixel values.
Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values 
that contain a start position and a run length.
E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels 1,2,3. The competition format requires a space
delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. 
The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. 
----------------------------------------------------------------------------------------------------------------
Put the predicted segmentation images into the "image" folder(You need to create one) and run this python file.
"""

def get_img_file(image_dir):
    imagelist = []
    namelist = []
    for parent, dirnames, filenames in os.walk(image_dir):
        for filename in filenames:
            if filename.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.gif')):
                imagelist.append(os.path.join(parent, filename))
                namelist.append(filename)
        return imagelist, namelist


def turn_to_str(image_list):
    outputs = []
    for image_path in image_list:
        image = Image.open(image_path).convert('L')
        transform = torchvision.transforms.ToTensor()
        image = image.resize((512, 512), Image.Resampling.BILINEAR)
        image = transform(image)
        image[image > 0] = 1
        dots = np.where(image.flatten() == 1)[0]
        run_lengths = []
        prev = -2
        for b in dots:
            if (b > prev + 1):
                run_lengths.extend((b + 1, 0))
            run_lengths[-1] += 1
            prev = b
        output = ' '.join([str(r) for r in run_lengths])
        outputs.append(output)
    return outputs


def save_to_csv(name_list, str_list):
    df = pd.DataFrame(columns=['Id', 'Predicted'])
    df['Id'] = [i.split('.')[0] for i in name_list]
    df['Predicted'] = str_list
    df.to_csv('submission.csv', index=None)


if __name__=="__main__":
    image_dir = './image'
    image_list, name_list = get_img_file(image_dir)
    str_list = turn_to_str(image_list)
    save_to_csv(name_list, str_list)
