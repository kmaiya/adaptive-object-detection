import matplotlib.pyplot as plt

"""
    This function is the code used to plot the detections of yolov3.
    The exact repo can still be found on the trello board.
  
    imgs:               list of image paths
    
    img_detections:     list of the detection results (default data structure is the output of YOLOv3)
        [x1, y1, x2, y2, object_confidence, class_confidence, class_score (class assignment)] 
    
    output_directory:   directory to save the images with predicted bounding boxes
    
    classes:            list of classes. In this function, we filter based on the index of where 'person'
                        is in the list. The default placement is based on the output of YOLOv3. 
                        Be careful if you use a class list from a different repo or algorithm, etc.
        ['person', 'cat', 'dog', etc]

    img_size:           size of the image.
"""
def plot_detections(imgs, img_detections, output_directory, classes, img_size):
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0,1,20)]
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        print ("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)



        """ I'm not sure if you need the padding if you aren't using YOLOv3 """
        # The amount of padding that was added
        pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
        
        """ I'm not sure if you need the padding if you aren't using YOLOv3 """
        # Image height and width after padding is removed
        unpad_h = img_size - pad_y
        unpad_w = img_size - pad_x

        # Draw bounding boxes and labels of detections
        if detections is not None:
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                
                '''If the class prediction is not a person, ignore it'''
                if not (classes[int(cls_pred)] == 'person'):
                    continue
               
                '''Just some printing for viewing predictions, etc'''
                print ('\t+ Label: %s, Object Conf: %.5f, Class Conf: %.5f' % (classes[int(cls_pred)], conf, cls_conf.item()))

                # Rescale coordinates to original dimensions
                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

                color = 'red'

                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                        edgecolor=color, facecolor=None)
 
                # Add the bbox to the plot
                ax.add_patch(bbox)
                
                ''' If we want to add text above the bounding box, uncomment the following lines.
                    This will add a string = class prediction + class score + objectness score
                    to the text above the bounding box.
                    
                    For now, lets leave it out...'''
                #label=classes[int(cls_pred)] + " " + str(round(cls_conf.item(), 3)) + " " + str(round(conf.item(), 3))
                #plt.text(x1, y1, s=label, color='white', verticalalignment='top',
                #        bbox={'color': color, 'pad': 0}, fontsize=16)

        # Save generated image with detections
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig(output_directory+'output_%d.png' % (img_i), bbox_inches='tight', pad_inches=0.0)
        plt.close()
