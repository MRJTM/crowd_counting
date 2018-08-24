"""
after 2-divide_clip_result.py, we have create a lot images,density_map,ground_truth folders
so we need to collect merge those folders into a big images,density_map and ground_truth

"""
import os
import sys
from multiprocessing import Process

def clone_files(source_path,des_path,file_names,file_type,block_index, block_size,global_start_index,density_type,sub_type):
    start=global_start_index+block_index*block_size

    for k,file_name in enumerate(file_names):
        # 打印进度
        finished = k / len(file_names) * 100
        print('{}/{}:process {} finished {:.3f}%'.format(density_type,sub_type,block_index + 1, finished))

        source_image_name=source_path+'/'+file_name
        des_image_name=des_path+'/'+'{:0>5d}'.format(start+k)+file_type
        os.system('sudo cp {} {}'.format(source_image_name,des_image_name))


def merge_dataset(dataset_path,save_path,num_process):

    # 先处理文件路径问题
    if not os.path.exists(dataset_path):
        print('{} does not exist'.format(dataset_path))
        exit(0)


    # 读取不同密度的四种文件夹，分别处理其中的images,density_map,ground_truth文件夹
    folders=['XL_density_divided','L_density_divided','M_density_divided','S_density_divided']

    for folder in folders:
        print('dealing with {}......................'.format(folder))

        # 目标文件夹路径
        final_image_path = save_path + '/'+ folder.split('_')[0]+'_'+folder.split('_')[1]+'/' + 'image'
        final_density_map_path = save_path + '/'+ folder.split('_')[0]+'_'+folder.split('_')[1]+'/'+ 'density_map'
        final_ground_truth_path = save_path + '/' +folder.split('_')[0]+'_'+folder.split('_')[1]+'/' 'ground_truth'

        final_paths = [save_path, final_image_path, final_density_map_path, final_ground_truth_path]

        for path in final_paths:
            if not os.path.exists(path):
                os.system('mkdir -p {}'.format(path))

        # 计算一个 start_id
        start_id = len(os.listdir(final_image_path))

        # 源文件夹路径
        image_path=dataset_path+'/'+folder+'/'+'images'
        density_map_path=dataset_path+'/'+folder+'/'+'density_map'
        ground_truth_path=dataset_path+'/'+folder+'/'+'ground_truth'
        file_paths=[image_path,density_map_path,ground_truth_path]

        # 对每个文件夹中的文件，使用多线程进行文件复制
        for f,file_path in enumerate(file_paths):

            print('\tdealing with {}.......'.format(file_path.split('/')[-1]))
            # 获取文件类型
            if f!=2:
                file_type='.jpg'
            else:
                file_type='.npy'

            # 根据进程数量来配置文件数
            total_file_list = os.listdir(file_path)
            total_file_list.sort()
            total_num = len(total_file_list)

            # 计算每个进程要处理的图片数量
            num_pic_per_process = round(total_num / num_process)
            num_pic_last_process = total_num - 7 * num_pic_per_process

            # 将文件分成与进程数相当的块数
            fileblock_list = []
            for i in range(num_process):
                if i < num_process-1:
                    start_index = i * num_pic_per_process
                    end_index = (i + 1) * num_pic_per_process
                else:
                    start_index = i * num_pic_per_process
                    end_index = start_index + num_pic_last_process

                file_block = total_file_list[start_index:end_index]
                fileblock_list.append(file_block)


            process = [Process(target=clone_files,
                               args=(file_path,final_paths[f+1],block,file_type,block_index,num_pic_per_process,
                                     start_id,folder.split('_')[0]+'_'+folder.split('_')[1],file_path.split('/')[-1]))
                        for block_index, block in enumerate(fileblock_list)]
            for p in process:
                p.start()
            for p in process:
                p.join()



"""----------------------------------执行程序------------------------------------------------------"""
dataset_path= sys.argv[1]
save_path='../dataset/final'
num_process=8

merge_dataset(dataset_path,save_path,num_process)