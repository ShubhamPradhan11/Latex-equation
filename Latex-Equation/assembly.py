####################################################################################################################################################################
#assembly

def get_control_chars():
    control=['alpha','beta','gamma','delta','epsilon','zeta','eta','theta','iota','kappa','lambda','mu','nu','xi','pi','rho','sigma','tau','upsilon','phi','chi','psi','omega','Alpha','Beta','Gamma','Delta','Epsilon','Zeta','Eta','Theta','Iota','Kappa','Lambda','Mu','Nu','Xi','Pi','Rho','Sigma','Tau','Upsilon','Phi','Psi','Omega','int','rightarrow','infty']
    return control

def get_det_char(detected,index, next_det,next_seg_dist,control,limit_control,space):
    # Recognize control characters
    #print(detected,next_seg_dist,space)
    if (detected in control) or (detected in limit_control):
        new_detected = '\\'+detected+' '
        detected = new_detected

    # Insert space if needed between letters
#     print(check_letter(detected),check_letter(next_det),next_det)
    if (next_seg_dist is not None) and (check_letter(detected)) and (check_letter(next_det)):
        if next_seg_dist >= space:
            detected = detected+'\,'

    return detected

def check_letter(chara):
    if len(chara)==1 and chara.isalpha():
        return True
    return False

def assemble_eqn(bboxs,centroids,detected_chars,flag_recur=0):
    
    control = get_control_chars()
    limit_control = ['int','sum','prod','lim'] #Controls that can have \limits in tex eqn 
    prev_centr_yco = None
    prev_fraction = False 
    super_exists = False # Flag to check if we are in an exponent
    sub_exists = False # Flag for subscript
    space = 12

    num_chars = centroids.shape[1]
    
    #boundingbox: Upper left x, Upper left y, width, height
    boxes = np.copy(bboxs)
    idxs = np.argsort(boxes[0,:])
    new_boxes = boxes[:,idxs]
    boxes = copy.deepcopy(new_boxes)

    new_centroids = centroids[:,idxs]
    new_detected_chars = [None]*centroids.shape[1]
    #print(detected_chars,idxs,centroids,'aaa')
    for i in range(0,centroids.shape[1]):
        new_detected_chars[i] = detected_chars[idxs[i]]
    #print(new_detected_chars,"herehere")
    #print(new_boxes,new_centroids)

    i=0

    # Initialize the equation string
    eq_string = ''

    #print(num_chars)
    while i<num_chars:

        #print(i,num_chars,new_detected_chars[i])
        detected = new_detected_chars[i]
    
        # Check for overlaps
        upper_left_x = boxes[0,i]
        upper_right_x = upper_left_x+boxes[2,i]

        #print(boxes[1,:])
        overlap_idx = np.logical_and(boxes[0,:]>=upper_left_x,boxes[0,:]<=upper_right_x)
        current_olap_boxes = boxes[:,overlap_idx]
        try:
            current_olap_top = np.min(current_olap_boxes[1,:])
            current_olap_bottom = np.max(current_olap_boxes[1,:]+current_olap_boxes[3,:])
            total_height = current_olap_bottom - current_olap_top
        except :
            total_height = 0
        overlap_idx = np.logical_and(np.logical_and(boxes[0,:]>=upper_left_x,boxes[0,:]<=upper_right_x),boxes[3,:] < 0.7*total_height)
        
        #get all overlaps
        current_olap_boxes = boxes[:,overlap_idx]
        upper_left_list = current_olap_boxes[0,:]
        upper_right_list = current_olap_boxes[0,:]+current_olap_boxes[2,:]
        top_list = current_olap_boxes[1,:]
        bottom_list = current_olap_boxes[1,:] + current_olap_boxes[3,:]

        while (len(upper_left_list)!=0) and ((np.min(upper_left_list) < upper_left_x) or (np.max(upper_right_list) > upper_right_x)):
            upper_left_x = np.min(upper_left_list)
            upper_right_x = np.max(upper_right_list)
            total_height = np.max(bottom_list)-np.min(top_list)
            overlap_idx = np.logical_and(np.logical_and(boxes[0,:]>=upper_left_x,boxes[0,:]<=upper_right_x),boxes[3,:] < 0.7*total_height)
            current_olap_boxes = boxes[:,overlap_idx]
            upper_left_list = current_olap_boxes[0,:]
            upper_right_list = current_olap_boxes[0,:]+current_olap_boxes[2,:]
            

        overlap_idx[i] = False
        num_overlaps = np.sum(overlap_idx)
        #print(overlap_idx)
        if detected == 'sqrt':
            #print("in squareroot")
            overlap_det_chars = []
            for k,w_char in zip(overlap_idx,new_detected_chars):
                if k:
                    overlap_det_chars.append(w_char)
            sqrt_str = assemble_eqn(boxes[:,overlap_idx],new_centroids[:,overlap_idx],overlap_det_chars)
            eq_string+='\sqrt{'+sqrt_str+'}'
            i = i + num_overlaps + 1
            continue
        if num_overlaps == 0:
            
             # Check for super and sub scripts by comparing lower left corner with stored centroid height, unless it was fraction
            if i > 0 and ~prev_fraction:
                
                lower_left_corn = boxes[1,i]+boxes[3,i]
                upper_left_corn = boxes[1,i]
                if lower_left_corn <= np.ceil(prev_centr_yco)+3:

                    if detected!='-' or lower_left_corn < (prev_centr_yco-5*boxes[3,i]):
                        # control sequence
                        super_exists = True
                        #print("I got power here 1")
                        new_eq_string = eq_string.strip() +'^{'
                        eq_string = new_eq_string
                        
                        prev_centr_yco = new_centroids[0,i]
                        
                        if i+1< num_chars:
                            next_seg_dist = boxes[0,i+1]-upper_right_x
                            next_send = new_detected_chars[i+1]
                        else:
                            next_seg_dist = None
                            next_send = None
                        
                        detected = get_det_char(detected,i,next_send,next_seg_dist,control, limit_control,space)
    
                elif upper_left_corn >= np.floor(prev_centr_yco):
                    #print(upper_left_corn,prev_centr_yco)
                 
                    if super_exists:
                        super_exists = False
                        # End of exponent
                        new_eq_string = eq_string.strip() +'}'
                        eq_string = new_eq_string
            
                        prev_fraction = False
                        
                        # Store centroid to check for exponents
                        prev_centr_yco = new_centroids[0,i]
                        
                        if i+1< num_chars:
                            next_seg_dist = boxes[0,i+1]-upper_right_x
                            next_send = new_detected_chars[i+1]
                        else:
                            next_seg_dist = None
                            next_send = None
                        
                        detected = get_det_char(detected,i,next_send,next_seg_dist,control, limit_control,space)
                    else:
                        # Subscript
                        sub_exists = True
                        new_eq_string = eq_string.strip() +'_{'
                        eq_string = new_eq_string
                        
                        if i+1< num_chars:
                            next_seg_dist = boxes[0,i+1]-upper_right_x
                            next_send = new_detected_chars[i+1]
                        else:
                            next_seg_dist = None
                            next_send = None

                        detected = get_det_char(detected,i,next_send,next_seg_dist,control, limit_control,space)
                        
                        # Store centroid to check for exponents
                        prev_centr_yco = new_centroids[0,i]
                else:
                    prev_fraction = False
                    
                    # Store centroid to check for exponents
                    prev_centr_yco = new_centroids[0,i]
                    
                    if i+1< num_chars:
                        next_seg_dist = boxes[0,i+1]-upper_right_x
                        next_send = new_detected_chars[i+1]
                    else:
                        next_seg_dist = None
                        next_send = None
                    
                    detected = get_det_char(detected,i,next_send,next_seg_dist,control, limit_control,space)
            else:
                prev_fraction = False;
                
                # Store centroid to check for exponents
                prev_centr_yco = new_centroids[0,i]
                
                if i+1< num_chars:
                    next_seg_dist = boxes[0,i+1]-upper_right_x
                    next_send = new_detected_chars[i+1]
                else:
                    next_seg_dist = None
                    next_send = None
                
                detected = get_det_char(detected,i,next_send,next_seg_dist,control, limit_control,space)

            eq_string = eq_string+detected
        elif num_overlaps==1:

            overlap_ul = boxes[0,i+1]
            if np.abs(overlap_ul-upper_right_x) <= 1:
                if detected in control:
                    new_detected = '\\'+detected+' '
                    detected = new_detected
                
                
                if i+1< num_chars and check_letter(detected) and check_letter(new_detected_chars[i+1]):
                    next_seg_dist = boxes[0,i+1]-upper_right_x
                    if next_seg_dist >= space:
                        detected = detected+'\,'
                
                eq_string = eq_string +detected
                
                
                # Store centroid to check for exponents
                prev_centr_yco = new_centroids[0,i]
            else:
                overlap_char = new_detected_chars[i+1]
                #print(overlap_char,detected)
                # Manually check for = case
                # If just a two '-', it is an equals
                if detected=='-' and overlap_char=='-':
                    eq_string = eq_string +'='
                    prev_centr_yco = 1/2*(new_centroids[0,i]+new_centroids[0,i+1])
                
                i = i+1 # Skip next char
        else:
            # Get the overlapped boxes, look for bars and limit controls
            overlap_idx[i] = True
            overlap_indi = []
            overlap_det_chars = []
            ct = 0
            for val in overlap_idx:
                if val:
                    overlap_indi.append(ct)
                    overlap_det_chars.append(new_detected_chars[ct])
                ct+=1
            
            overlap_centroids = np.zeros((2,len(overlap_indi)))
            overlap_boxes = np.zeros((4,len(overlap_indi)))
            bar_idx = []
            bar_width = []
            limit_idx = []
            limit_height = []
            overlap_centroids = new_centroids[:,overlap_idx]
            overlap_boxes = boxes[:,overlap_idx]
            for overlap_i in range(0,len(overlap_indi)):
#                 overlap_centroids[:,overlap_i] = new_centroids[:,overlap_indi[overlap_i]]
#                 overlap_boxes[:,overlap_i] = boxes[:,overlap_indi[overlap_i]]
                if new_detected_chars[overlap_indi[overlap_i]]=='-':
                    bar_idx.append(overlap_i)
                    bar_width.append(boxes[2,overlap_indi[overlap_i]])
                elif new_detected_chars[overlap_indi[overlap_i]] in limit_control:
                    limit_idx.append(overlap_i)
                    limit_height.append(boxes[3,overlap_indi[overlap_i]])
            
            upper_left_list = overlap_boxes[0,:]
            upper_right_list = overlap_boxes[0,:]+overlap_boxes[2,:]
            total_width = np.max(upper_right_list)-np.min(upper_left_list)
            
            bar_idx = np.array(bar_idx)
            bar_width = np.array(bar_width)
            bar_idx = bar_idx[bar_width >= 0.8*total_width]
            
            # If we have a single fraction bar recognized
            if bar_idx.size == 1:
                prev_fraction = True;
                frac_y_coord = overlap_centroids[0,bar_idx]
                frac_bar_height = overlap_boxes[3,bar_idx]
                
               
                #print(new_detected_chars[i])
                if (prev_centr_yco is not None) and (frac_y_coord < prev_centr_yco - 7*frac_bar_height):
                    super_exists = True
                    #print("I got power here 3")
                    new_eq_string = eq_string.strip()+ '^{'
                    eq_string = new_eq_string
                if prev_centr_yco is None: 
                    prev_centr_yco = frac_y_coord;
                
                # Get the numerator and denominator characters by comparing centroid to the fraction bar's y co
                num_idx = overlap_centroids < frac_y_coord
                num_idx = num_idx[0,:]
                denom_idx = overlap_centroids > frac_y_coord
                denom_idx = denom_idx[0,:]
                num_det_chars = []  
                for k,w_char in zip(num_idx,overlap_det_chars):
                    if k:
                        num_det_chars.append(w_char)
                denom_det_chars = []  
                for k,w_char in zip(denom_idx,overlap_det_chars):
                    if k:
                        denom_det_chars.append(w_char) 

                # Get eq_string of subequation string
                numer_str = assemble_eqn(overlap_boxes[:,num_idx],overlap_centroids[:,num_idx],num_det_chars)
                denom_str = assemble_eqn(overlap_boxes[:,denom_idx],overlap_centroids[:,denom_idx],denom_det_chars)
                detected = '\\frac{'+numer_str+'}{'+denom_str+'}'
                
                #  fraction must be end of exponent
                if super_exists:
                    super_exists = False;
                    detected = detected +'}'
            
                
                # Set counter to next character that is not in fraction
                i = np.argwhere(overlap_idx==True)[-1][0]
            
            elif len(limit_idx)!=0:
                dom_idx = np.argmax(np.array(limit_height)) # Choose largest to be the limit_char
                dom_idx = limit_idx[dom_idx]
                #print(dom_idx,overlap_det_chars,overlap_boxes.size)
                limit_char_detected = overlap_det_chars[dom_idx]
                limit_y_coord = overlap_centroids[0,dom_idx]
                
                prev_centr_yco = limit_y_coord;
                
                # Get the top and bottom characters by comparing centroid to the fraction bar's y co
                top_idx = overlap_centroids < limit_y_coord;
                top_idx = top_idx[0,:]
                bottom_idx = overlap_centroids > limit_y_coord;
                bottom_idx = bottom_idx[0,:]
                
                detected = '\\' + limit_char_detected + '\limits'
                if np.any(bottom_idx):
                    bottom_det_chars = []  
                    for k,w_char in zip(bottom_idx,overlap_det_chars):
                        if k:
                            bottom_det_chars.append(w_char)
                    bottom_str = assemble_eqn(overlap_boxes[:,bottom_idx],overlap_centroids[:,bottom_idx],bottom_det_chars)
                    detected +=  '_{' +bottom_str +'}'
                if np.any(top_idx):
                    top_det_chars = []  
                    for k,w_char in zip(top_idx,overlap_det_chars):
                        if k:
                            top_det_chars.append(w_char)
                    top_str = assemble_eqn(overlap_boxes[:,top_idx],overlap_centroids[:,top_idx],top_det_chars)
                    #print("I got power here 2")
                    detected +=  '^{' +top_str +'}'
                
                
                # Set i to next character
                i = np.argwhere(overlap_idx==True)[-1][0]
            
            
            eq_string = eq_string+ detected
            
        #print("near i+1",eq_string)
        i+=1
    if super_exists or sub_exists:
        eq_string = eq_string.strip()+'}'
    return eq_string

def get_boxes(bboxes):
    boxes = np.zeros((4,len(bboxes)))
    for i in range(0,len(bboxes)):
        boxes[0,i] = bboxes[i][1]
        boxes[1,i] = bboxes[i][0]
        boxes[2,i] = bboxes[i][3] - bboxes[i][1]
        boxes[3,i] = bboxes[i][2] - bboxes[i][0]
    return boxes

def get_centroids(centroids):
    centrs = np.zeros((2,len(centroids)))
    for i in range(0,len(centroids)):
        centrs[0,i] = centroids[i][0]
        centrs[1,i] = centroids[i][1]
    return centrs