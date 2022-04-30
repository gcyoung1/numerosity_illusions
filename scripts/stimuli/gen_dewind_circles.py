'''

This dataset generator generates binary images of equally sized white circles on black backgrounds. 

Following Dewind et al 2015, there are three orthogonal dimensions along which the generated images differ (placement of the dots in the image notwihstanding): log(numerosity), where numerosity is the number of dots in the image; log(Size), where Size = ISA*TSA, or equivalently n*ISA^2; and log(Spacing), where Spacing = FA*Sparsity, or equivalently 1/n * FA^2; where ISA (Individual Surface Area) is the surface area of each individual dot in the image, TSA (Total Surface Area) is n*ISA, FA (Field Area) is the total background area within which all dots are drawn, and Sparsity is FA/n.

By default, stimuli are generated by specifying a list of values for each of the three orthogonal dimensions, as well as the number of images to make for every combination of these three dimensions. The --linear flag can be used to specify numerosity, Size, and Spacing exactly instead of their logs. From there, ISA and FA are calculated for each combination of dimensions and stimuli are generated by randomly choosing a circular region in the image with the calculated FA and then using rejection sampling to generate the right number of dots in that region, each with the calculated ISA. That is, dots are generated sequentially such that the distance between any two dots (or any dot and the edge of the image) is less than the --distance argument. The dimensions of the image in pixels can be specified with the --pic_dim argument.

File names of images, placed in a directory of the /stimuli/ directory, follow the following format:
[numerosity]_[Size]_[Spacing]_[unique image #].png

'''


from PIL import Image, ImageDraw
import numpy as np
import os
import time
import argparse
import math
from geometry_utils import polar_to_cartesian, radius_from_area
from circle import Circle

def gen_circle_in_field(field, individual_radius, min_distance):
    r = np.random.uniform(0,field.radius-individual_radius)
    theta = np.random.uniform(0,360)
    individual_center = field.center + polar_to_cartesian(r, theta)
    return Circle(individual_center, individual_radius)

def gen_circles(numerosity, individual_radius, field, min_distance, pic_width, pic_height):
    # NB This will infinite loop if the input parameters are impossible to satisfy
    while True:
        circles = []
        for _ in range(numerosity):
            # Try to generate a new circle 200 times
            for attempt in range(200):
                circle = gen_circle_in_field(field, individual_radius, min_distance)
                untouched = True
                for other_circle in circles:
                    if circle.distance_from(other_circle) < min_distance:
                        untouched = False
                # Break out early if you succeed
                if untouched: 
                    break
            # If you didn't succeed in time, go back to the beginning and try again
            if attempt == 199:
                break
            # Otherwise add to your list of circles and return if you have all you need
            circles.append(circle)
            if len(circles) == numerosity:
                return circles

def gen_circle_in_rectangle(x_center,y_center,rect_width,rect_height,radius):
    max_x_change = (rect_width/2) - radius
    max_y_change = (rect_height/2) - radius
    x = np.random.uniform(x_center-max_x_change,x_center+max_x_change)
    y = np.random.uniform(y_center-max_y_change,y_center+max_y_change)
    center = np.array([x,y])
    return Circle(center, radius)

def gen_image(numerosity, size, spacing, min_distance, pic_width, pic_height):
    individual_surface_area = (size/numerosity)**(1/2)
    individual_radius = radius_from_area(individual_surface_area)
    field_area = (spacing*numerosity)**(1/2)
    field_radius = radius_from_area(field_area)
    field = gen_circle_in_rectangle(pic_width/2,pic_height/2,pic_width,pic_height,field_radius)

    img = Image.new('1', (pic_width, pic_height), 'black')
    circles = gen_circles(numerosity, individual_radius, field, args.min_distance, pic_width, pic_height)
    for circle in circles:
        corners = circle.corners()
        circledraw = ImageDraw.Draw(img)
        fill_color = 'white'
        circledraw.ellipse(corners, fill=fill_color, outline='white')

    return img

if __name__ == '__main__':

    start_time = time.time()

    #Command Line Arguments 
    parser = argparse.ArgumentParser(description='Generate Dewind stimuli')
    parser.add_argument('--dataset_name', type=str, help='Name of dataset directory.')
    parser.add_argument('--pic_width', type=int, default=100, help='number of pixels for width of image. Default = 100')
    parser.add_argument('--pic_height', type=int, default=100, help='number of pixels for height of image. Default = 100')
    parser.add_argument('--linear_args', action='store_true', default=False, help="If this argument is used, interpret numerosities, sizes, and spacings linearly. Otherwise, assume they are log_2 of the actual desired values")
    parser.add_argument('--numerosities', nargs='+', type=int, help='space separated list of the number of dots. Log_2 scaled by default: use the --linear_args argument to interpret linearly.')
    parser.add_argument('--sizes', nargs='+', type=int, help='space separated list of the Sizes. Log_2 scaled by default: use the --linear_args argument to interpret linearly.')
    parser.add_argument('--spacings', nargs='+', type=int, help='space separated list of the Spacings. Log_2 scaled by default: use the --linear_args argument to interpret linearly.')
    parser.add_argument('--min_distance', type=int, default=1, help='minimum number of pixels between the edges of each dot and between the edge of each dot and the edge of the image. Default = 1.')
    parser.add_argument('--num_pics_per_category', type=int, 
                        help='number of pictures per combination of stimulus parameters')
    parser.add_argument('--num_train_pics_per_category', type=int,
                        help='number of training pictures per combination of stimulus parameters')
    

    args = parser.parse_args()
    # reconcile arguments
    if not args.linear_args:
        def square_all(l):
            return [2**x for x in l]
        args.numerosities = square_all(args.numerosities)
        args.sizes = square_all(args.sizes)
        args.spacings = square_all(args.spacings)
    if args.num_train_pics_per_category > args.num_pics_per_category:
        raise ValueError("Can't have more train pics than total pics.")

    dataset_name = args.dataset_name+'_dewind_circles_'+time.strftime('%m-%d-%Y:%H_%M')
    outputdir = os.path.join('../../data/stimuli',dataset_name)    
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)


    os.mkdir(os.path.join(outputdir,'train'))
    train_dir = os.path.join(outputdir,'train')
    os.mkdir(os.path.join(outputdir,'test'))
    test_dir = os.path.join(outputdir,'test')

    with open(os.path.join(outputdir,'args.txt'), 'w') as argsfile:
        argsfile.write(str(args))
    print('running with args:')
    print(args)

    for numerosity in args.numerosities:
        print(numerosity)
        for size in args.sizes:
            print(f"Size: {size}")
            for spacing in args.spacings:
                print(f"Spacing: {spacing}")
                for pic_index in range(1,args.num_pics_per_category):
                    img = gen_image(numerosity, size, spacing, args.min_distance, args.pic_width, args.pic_height)
                    img_file_name = f"{numerosity}_{size}_{spacing}_{pic_index}.png"
                    if pic_index <= args.num_train_pics_per_category:
                        img.save(os.path.join(train_dir,img_file_name))
                    else:
                        img.save(os.path.join(test_dir,img_file_name))


    end_time = time.time()
    print('Run Time: %s'%(end_time-start_time))

