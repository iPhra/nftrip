import os
import subprocess
from shutil import copyfile


def copy_images(results_path):
    results = list(sorted(results_path.glob('*.jpg')))
    total_images = len(results)

    new_folder = results_path.parent / (results_path.name + '_new')
    new_folder.mkdir(exist_ok=True, parents=True)

    to_keep = []
    for i in range(10):
        to_keep += [results[i]]*2

    for i in range(10, total_images//10, 10):
        to_keep += [results[i]]
        
    for i in range(total_images//10, total_images, 50):
        to_keep += [results[i]]

    to_keep += [to_keep[-1]]*10
    to_keep = to_keep + to_keep[::-1]

    for i, file in enumerate(to_keep):
        new_name = f'{i:04d}'
        copyfile(file, new_folder/f'{new_name}.jpg')
    
    return new_folder



def create_video_from_intermediate_results(results_path, img_format):
    import shutil
    #
    # change this depending on what you want to accomplish (modify out video name, change fps and trim video)
    #
    out_file_name = 'out.mp4'
    fps = 30
    first_frame = 0
    number_of_frames_to_process = len(os.listdir(results_path))  # default don't trim take process every frame

    ffmpeg = 'ffmpeg'
    if shutil.which(ffmpeg):  # if ffmpeg is in system path
        img_name_format = '%' + str(img_format[0]) + 'd' + img_format[1]  # example: '%4d.png' for (4, '.png')
        pattern = os.path.join(results_path, img_name_format)
        out_video_path = os.path.join(results_path, out_file_name)

        print('Creating video..')
        trim_video_command = ['-start_number', str(first_frame), '-vframes', str(number_of_frames_to_process)]
        input_options = ['-r', str(fps), '-i', pattern, '-y']
        encoding_options = ['-c:v', 'libx264', '-crf', '25', '-pix_fmt', 'yuv420p', '-vf', "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
        subprocess.call([ffmpeg, *input_options, *trim_video_command, *encoding_options, out_video_path])

        print('Creating gif..')
        output_gif_path = results_path / 'out.gif'
        input_options = ['-ss', "1", '-t', '10', '-i', out_video_path, '-vf', 'fps=30,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse', '-loop', '0', str(output_gif_path)]
        subprocess.call([ffmpeg, *input_options])
    else:
        print(f'{ffmpeg} not found in the system path, aborting.')

