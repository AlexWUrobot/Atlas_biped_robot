#!/usr/bin/env python
import rospy
from gazebo_msgs.msg import LinkStates
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

from pyquaternion import Quaternion
from math import cos, sin, acos, asin, atan, pi

#def global_fun(void)
#    global x


#import robot_controller_v2


bool_time = True
#bool_goal = True



def callback_torque(data):
    # l_leg_kny, l_leg_lhy, l_leg_uay, r_leg_kny, r_leg_lhy, r_leg_uay    # 0~5  totally six links 
    global r_ankle_torque
    rospy.loginfo("callback r_leg_uay torque: %s", data.effort[5])
    r_ankle_torque = data.effort[5];
    #rospy.loginfo("Get Torque !!!!!!!!!")


def callback(data): 
    #rospy.loginfo("I heard %s", data.pose[1].position.x)
    #rospy.loginfo("Yes============ %f ", x)  # sometime it cannot get it  
    global CoM_cur
    global CoM_last
    global velocity
    len_link = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #global len_link[6] why can not pass



    # global coordinate from Linkstate    position.x // orientation  xyzw
    pos_arr = { 'dummy':data.pose[4], 'l_uleg':data.pose[5], 'l_lleg':data.pose[6], 'l_talus':data.pose[7], 'r_uleg':data.pose[8], 'r_lleg':data.pose[9], 'r_talus':data.pose[10] }


 #['ground_plane::link', 'atlas_bipedal::fixed_base', 'atlas_bipedal::dummy_rev', 'atlas_bipedal::dummy_pris',
 # 'atlas_bipedal::pelvis', 'atlas_bipedal::l_uleg', 'atlas_bipedal::l_lleg', 'atlas_bipedal::l_talus',
 # 'atlas_bipedal::r_uleg', 'atlas_bipedal::r_lleg', 'atlas_bipedal::r_talus']


    x1 = pos_arr['dummy'].position.x
    y1 = pos_arr['dummy'].position.y
    z1 = pos_arr['dummy'].position.z

    x2 = pos_arr['l_uleg'].position.x
    y2 = pos_arr['l_uleg'].position.y
    z2 = pos_arr['l_uleg'].position.z
    len_link[0] = ( (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2 )**(0.5)

    x1 = pos_arr['l_uleg'].position.x
    y1 = pos_arr['l_uleg'].position.y
    z1 = pos_arr['l_uleg'].position.z

    x2 = pos_arr['l_lleg'].position.x
    y2 = pos_arr['l_lleg'].position.y
    z2 = pos_arr['l_lleg'].position.z
    len_link[1] = ( (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2 )**(0.5)

    x1 = pos_arr['l_talus'].position.x
    y1 = pos_arr['l_talus'].position.y
    z1 = pos_arr['l_talus'].position.z

    x2 = pos_arr['l_lleg'].position.x
    y2 = pos_arr['l_lleg'].position.y
    z2 = pos_arr['l_lleg'].position.z
    len_link[2] = ( (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2 )**(0.5)

    #==========================================
    x1 = pos_arr['dummy'].position.x
    y1 = pos_arr['dummy'].position.y
    z1 = pos_arr['dummy'].position.z

    x2 = pos_arr['r_uleg'].position.x
    y2 = pos_arr['r_uleg'].position.y
    z2 = pos_arr['r_uleg'].position.z
    len_link[3] = ( (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2 )**(0.5)

    x1 = pos_arr['r_uleg'].position.x
    y1 = pos_arr['r_uleg'].position.y
    z1 = pos_arr['r_uleg'].position.z

    x2 = pos_arr['r_lleg'].position.x
    y2 = pos_arr['r_lleg'].position.y
    z2 = pos_arr['r_lleg'].position.z
    len_link[4] = ( (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2 )**(0.5)

    x1 = pos_arr['r_talus'].position.x
    y1 = pos_arr['r_talus'].position.y
    z1 = pos_arr['r_talus'].position.z

    x2 = pos_arr['r_lleg'].position.x
    y2 = pos_arr['r_lleg'].position.y
    z2 = pos_arr['r_lleg'].position.z
    len_link[5] = ( (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2 )**(0.5)

#pos_num = { '0':data.pose[4], '1':data.pose[5], '2':data.pose[6], '3':data.pose[7], '4':data.pose[8], '5':data.pose[9], '6':data.pose[10] }

    # link length should be same
#    for i in range(0,3):  # 0~2
#        x1 = pos_num[i].position.x
#        y1 = pos_num[i].position.y
#        z1 = pos_num[i].position.z

#        x2 = pos_num[i+1].position.x
#        y2 = pos_num[i+1].position.y
#        z2 = pos_num[i+1].position.z
#        len_link[i] = ( (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2 )**(0.5)

    # change the 'l_talus' from 7 to 4
#    pos_arr = { 'dummy':data.pose[4], 'l_uleg':data.pose[5], 'l_lleg':data.pose[6], 'l_talus':data.pose[4], 'r_uleg':data.pose[8], 'r_lleg':data.pose[9], 'r_talus':data.pose[10] }

#    for i in range(3,6):  # 3~5
#        x1 = pos_arr[i].position.x
#        y1 = data.pose[i].position.y
#        z1 = data.pose[i].position.z

#        x2 = data.pose[i+1].position.x
#        y2 = data.pose[i+1].position.y
#        z2 = data.pose[i+1].position.z
#        len_link[i] = ( (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2 )**(0.5)


    xy_ankle = [data.pose[7].position.x, data.pose[7].position.y]




    dt = 0.001;
    last_time = rospy.get_time()
    #curr_time = rospy.get_rostime()   # cannot work

    if bool_time == True:
	#rospy.loginfo(bool_time)
	global bool_time
        bool_time = False
        last_time = rospy.get_time()
    else:
 	#rospy.loginfo(bool_time)
	global bool_time
        bool_time = True
        curr_time = rospy.get_time()
        time_span = curr_time - last_time



    # Dummy 
    #q = Quaternion( pos_arr[0].orientation.x, 1, 0, 0)
    #q = Quaternion( data.pose[0].orientation.x, 1, 0, 0)   wxyz

    try:
	#rospy.loginfo("Yes============ %f ", x)
	
        CoM_cur = Calculate_CoM(pos_arr)
        rospy.loginfo("CoM: %f, %f, %f ", CoM_cur[0], CoM_cur[1], CoM_cur[2])
	#rospy.loginfo("TIME: %f, %f, %f", time_span, curr_time, last_time)  # why I cannot get the time span 




        if bool_time == True:
		CoM_last = [CoM_cur[0], CoM_cur[1], CoM_cur[2]]
	else:
		temp_dx = CoM_cur[0]-CoM_last[0];
		if temp_dx == 0:
			velocity = [0,0,0]
		else:
			velocity = [CoM_cur[0]-CoM_last[0],CoM_cur[1]-CoM_last[1], CoM_cur[2]-CoM_last[2]]
			velocity = velocity/dt;
	rospy.loginfo("Speed: %f, %f, %f", velocity[0], velocity[1], velocity[2])

        rospy.loginfo("link L: %f, %f, %f, %f",len_link[0],len_link[1],len_link[2],pos_arr['l_talus'].position.z)
        rospy.loginfo("link R: %f, %f, %f", len_link[3],len_link[4],len_link[5])
        #rospy.loginfo("r_uleg z: %f = %f ", pos_arr['l_uleg'].position.z ,len_link[1]+len_link[2]+pos_arr['l_talus'].position.z)

        # link L: 0.113670, 0.377327, 0.422000
        # link R: 0.113665, 0.377327, 0.422000
        # foot height: 0.880371 -- > 0.480371


        #=========================================  for walking
        global curr_time
        dt = 0.9
        interval = 2*pi/dt

        foot_height = 0.860371
        lift_height = 0.1


        if curr_time < 5:
            L_foot_goal_z = -foot_height
            R_foot_goal_z = -foot_height
        else:
            if sin(interval*curr_time) > 0:
                L_foot_goal_z = -foot_height + lift_height*sin(interval*curr_time)
                R_foot_goal_z = -foot_height
            else:
                R_foot_goal_z = -foot_height + lift_height*sin(interval*curr_time+pi)
                L_foot_goal_z = -foot_height

        curr_time = curr_time + 1

        rospy.loginfo("Succes v22")
        rospy.loginfo(L_foot_goal_z)
        rospy.loginfo(R_foot_goal_z)

        L_foot_goal = [0,0,L_foot_goal_z]
        R_foot_goal = [0,0,R_foot_goal_z]
        rospy.loginfo(L_foot_goal)
        rospy.loginfo(R_foot_goal)

#        L_foot_goal = [0,0,-0.780371]
#        R_foot_goal = [0,0,-0.780371]



        Lq = Inv(L_foot_goal)
        Rq = Inv(R_foot_goal)

        rospy.loginfo("Succes V55")



        #rospy.loginfo(Lq)  # q[2]~q[4]
        rospy.loginfo("q OK ")

        cmd_pos = [Lq[2],  Lq[3],  Lq[4],  Rq[2],  Rq[3],  Rq[4]]
        #===================================================== walking end

#        if bool_goal == true:
#        time_ach = 5000.0
#        span_2 = Lq[2]/time_ach
#        span_3 = Lq[3]/time_ach
#        span_4 = Lq[4]/time_ach

#        Rspan_2 = Rq[2]/time_ach
#        Rspan_3 = Rq[3]/time_ach
#        Rspan_4 = Rq[4]/time_ach

        #rospy.loginfo("Succes")


#        global cur_2, cur_3, cur_4
#        #rospy.loginfo("Succes V1.1")
#        global Rcur_2, Rcur_3, Rcur_4
#        #rospy.loginfo("Succes V1.2")

#        close_dis = abs(cur_2 - Rq[2])
#        #rospy.loginfo("Succes V1.5")

#        if close_dis > 0.005:
#            cur_2 = cur_2 + span_2
#            cur_3 = cur_3 + span_3
#            cur_4 = cur_4 + span_4
#            Rcur_2 = Rcur_2 + Rspan_2
#            Rcur_3 = Rcur_3 + Rspan_3
#            Rcur_4 = Rcur_4 + Rspan_4

#            rospy.loginfo("Succes V2")
#        cmd_pos = [cur_2,  cur_3,  cur_4,  Rcur_2,  Rcur_3,  Rcur_4]

        #cmd_pos = [-pi/6.0,  pi/2.0,  -pi/6.0,  0,  0,  0]   # positive left to back    pi/2 is the 90 degree by radian

        #cmd_pos = [q[2],  q[3],  q[4],  0,  0,  0]
        #cmd_pos = [0,  0,  0,  0,  0,  0]
                  #L hip knee ankle R knee,hip,ankle
        cmd_publisher(cmd_pos)  # X
        #rospy.loginfo([0, 0, cur_2,  cur_3,  cur_4, 0])
        rospy.loginfo("curr OK ")

        #rospy.loginfo(cmd_pos)


	CoP_cur = Cal_CoP(xy_ankle)
        #rospy.loginfo("CoP: %f, %f", CoP_cur[0], CoP_cur[1])
	#rospy.loginfo(q)
	#rospy.loginfo("Finish Calculate")
    except:
	return


def Cal_CoP(pos_arr):
    total_mass = 70.66
    z0 = CoM_cur[2]
    g = 9.81
    w0 = (g/z0)**(0.5)
    x_pos = 0
    y_pos = 0
    z_pos = 0
    dimenionless_torque = r_ankle_torque/ (total_mass*(w0**2)*(z0**2))
    x_pos = pos_arr[0] - dimenionless_torque
    y_pos = pos_arr[1]  
    xyz_cop = [x_pos, y_pos, 0]  
    return xyz_cop


def Cal_ICP(pos_vel_arr): # Input four factors include xy position and velocity
   w0 = (g/z0)**(0.5)

   x_pos = pos_vel_arr[0] + pos_vel_arr[3]/w0
   y_pos = pos_vel_arr[1] + pos_vel_arr[4]/w0

   xy_icp = []
   return xy_icp





def Calculate_CoM(pos_arr):

    # relativ_pos  relative to the joint
    relativ_pos = { 'dummy': [0.012957, -0.000586, 0.23336], 'l_uleg':[0, 0, -0.21], 'l_lleg':[0.001, 0, -0.187], 'l_talus':[0.025443, 0, -0.0631316], 'r_uleg':[0, 0, -0.21], 'r_lleg':[0.001, 0, -0.187], 'r_talus':[0.025443, 0, -0.0631316] }

    mass = [43.78,   7.34,  4.37,    1.73,   7.34,    4.37,  1.73]     # 7 links 1~ 7  0 = ground 
    x_pos = 0
    y_pos = 0
    z_pos = 0
    mass_tol = 0
    k = 0
    for i in pos_arr:   # 0~6 xrange(0, 7)
	#rospy.loginfo("i---------------%s ", i)  
	#rospy.loginfo("k--------------- %d ", k)  # 0~6

	q = Quaternion( pos_arr[i].orientation.w, pos_arr[i].orientation.x, pos_arr[i].orientation.y, pos_arr[i].orientation.z)  # w x y z  
	# http://kieranwynn.github.io/pyquaternion/#basic-usage
	#inv_q = q	               # 0.023580, -0.014518, 0.834510 
	inv_q = q.inverse              # 0.024223, -0.014503, 0.834514
	g_pos = inv_q.rotate(relativ_pos[i])  # turn to the global coordinate
	g_pos_x = g_pos[0] + pos_arr[i].position.x  # add the translation 
	g_pos_y = g_pos[1] + pos_arr[i].position.y
	g_pos_z = g_pos[2] + pos_arr[i].position.z   # //CoM xyz:0.0121242 -0.00223872 0.794455    q v conj 


	x_pos += g_pos_x*mass[k]
	y_pos += g_pos_y*mass[k]
	z_pos += g_pos_z*mass[k]

	mass_tol += mass[k]
	#x_pos += float(pos_arr[i].position.x)*mass[k]
	#y_pos += float(pos_arr[i].position.y)*mass[k]
	#z_pos += float(pos_arr[i].position.z)*mass[k]
	k += 1
    xyz_com = [x_pos/mass_tol, y_pos/mass_tol, z_pos/mass_tol]   # this can not   work ???? 
    #rospy.loginfo("CoM: %f, %f, %f ", xyz_com[0], xyz_com[1], xyz_com[2])
    return xyz_com
    #return xyz_com[x_pos/mass_tol, y_pos/mass_tol, z_pos/mass_tol] 
    # new model :  CoM: 0.019004, 0.023635, 2.233842



####################################################################




## Link states
#    def callback_linkstates(

#        link_states = data
#        # Calculate COM position
#        com_pos = array([0,0,0])
#        #com_pos = [0,0,0]

#        for i in range(7):

#            offset = offsets[i,:]
#            mass = masses[i]

#            q = link_states.pose[i+4].orientation
#            t = link_states.pose[i+4].position

#            quaternion = (q.x,q.y,q.z,q.w)
#            v_rot = qv_mult(quaternion,offset)

#            v_rot = np.array([v_rot[0]+t.x, v_rot[1] + t.y, v_rot[2] + t.z])

#            com_pos = com_pos + mass*v_rot

#        com_pos = com_pos/np.sum(masses)

#        # Calculate COM velocity frequency = 1000Hz
#        if np.sum(com_pre) == 0:
#            com_pre = com_pos
#            vel_pre = np.array([0,0,0])
#            com = np.append(com_pos,vel_pre)
#        else:
#            com_vel = (com_pos - com_pre)/ 0.001 # 1000 Hz
#            com = np.append(com_pos, com_vel)
#            com_pre = com_pos

#    # rotate vector v1 by quaternion q1
#    def qv_mult(quaternion,vector):

#        q1_conj = np.array((-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]), dtype=np.float64)

#        q2 = list(vector)
#        q2.append(0.0)

#        results = quat_mul( quat_mul(quaternion,q2), q1_conj )

#        return results[:3]

#    def quat_mul(quaternion1, quaternion0):

#        x0, y0, z0, w0 = quaternion0
#        x1, y1, z1, w1 = quaternion1

#        return np.array((x1*w0 + y1*z0 - z1*y0 + w1*x0,-x1*z0 + y1*w0 + z1*x0 + w1*y0,x1*y0 - y1*x0 + z1*w0 + w1*z0,-x1*x0 - y1*y0 - z1*z0 + w1*w0), dtype=np.float64)








# Publishers
global l_leg_lhy_pub, l_leg_kny_pub, l_leg_uay_pub, r_leg_lhy_pub, r_leg_kny_pub, r_leg_uay_pub

global i_error

# Initialize all torque command publisher
def pub_initial():

    global l_leg_lhy_pub, l_leg_kny_pub, l_leg_uay_pub, r_leg_lhy_pub, r_leg_kny_pub, r_leg_uay_pub

    l_leg_lhy_pub = rospy.Publisher('atlas_bipedal/l_leg_lhy_controller/command', Float64, queue_size=10, latch = True)
    l_leg_kny_pub = rospy.Publisher('atlas_bipedal/l_leg_kny_controller/command', Float64, queue_size=10, latch = True)
    l_leg_uay_pub = rospy.Publisher('atlas_bipedal/l_leg_uay_controller/command', Float64, queue_size=10, latch = True)
    r_leg_lhy_pub = rospy.Publisher('atlas_bipedal/r_leg_lhy_controller/command', Float64, queue_size=10, latch = True)
    r_leg_kny_pub = rospy.Publisher('atlas_bipedal/r_leg_kny_controller/command', Float64, queue_size=10, latch = True)
    r_leg_uay_pub = rospy.Publisher('atlas_bipedal/r_leg_uay_controller/command', Float64, queue_size=10, latch = True)


# Publish torque commands
def cmd_publisher(torque):

    global l_leg_lhy_pub, l_leg_kny_pub, l_leg_uay_pub, r_leg_lhy_pub, r_leg_kny_pub, r_leg_uay_pub

    # Publishing command
    l_leg_lhy_pub.publish(torque[0])             # actually it is position python robot_controller.py
    l_leg_kny_pub.publish(torque[1])            # how to know the position control
    l_leg_uay_pub.publish(torque[2])
    r_leg_lhy_pub.publish(torque[3])
    r_leg_kny_pub.publish(torque[4])
    r_leg_uay_pub.publish(torque[5])


# Inverse Kinematics Input xyz Output 6 angles
# foot_goal[x y z] compared to the 
def Inv(foot_goal):



    # 0.113670, 0.377327, 0.422000 0.081026
    Px = foot_goal[0]
    Py = foot_goal[1]
    Pz = foot_goal[2]
    l1 = 0
    l2 = 0
    l3 = 0.377327
    l4 = 0.422000
    l5 = 0
    l6 = 0.081026


    d2 = 0

    #ra = 0.0

    ra = 0.001

    R = [[cos(ra), 0, sin(ra)],[0, 1, 0],[-sin(ra), 0, cos(ra)]]

    r11 = R[0][0]
    r12 = R[0][1]
    r13 = R[0][2]
    r21 = R[1][0]
    r22 = R[1][1]
    r23 = R[1][2]
    r31 = R[2][0]
    r32 = R[2][1]
    r33 = R[2][2]


    q1 = 0
    q2 = 0
    q6 = 0


    H04x = Px + r13*(l6 + l5*cos(q6)) + l5*r12*sin(q6);
    H04y = Py + r23*(l6 + l5*cos(q6)) + l5*r22*sin(q6);
    H04z = Pz + r33*(l6 + l5*cos(q6)) + l5*r32*sin(q6);
    H02x = d2;
    H02y = l2*sin(q2);
    H02z = -l2*cos(q2);
    l24 = ((H04x-H02x)**2 + (H04y-H02y)**2+ (H04z-H02z)**2)**(.5)


    q4 = acos((l24**2-l3**2-l4**2) / (2*l3*l4))

    alpha = Py*sin(q2) - Pz*cos(q2) - (l6 + l5*cos(q6))*(r33*cos(q2) - r23*sin(q2)) - l2 - l5*sin(q6)*(r32*cos(q2) - r22*sin(q2))

    Be = d2 - Px - r13*(l6 + l5*cos(q6)) - l5*r12*sin(q6)
    ga = l3 + l4*cos(q4)


    phi = l3*ga + (l4**2)*(sin(q4)**2) + l4*cos(q4)*ga


    q3 = asin((Be*ga - alpha*l4*sin(q4)) / phi)

    A = cos(q6)*(r33*cos(q2) - r23*sin(q2)) + sin(q6)*(r32*cos(q2) - r22*sin(q2))
    B = r13*cos(q6) + r12*sin(q6)
    q5 = asin(B*cos(q3+q4) - A*sin(q3+q4))

    #Only use   q[2] q[3] q[4]
    q = [q1, q2, q3, q4, q5, q6]
    return q




def listener():
    #global_fun()
    #global l_leg_lhy_pub, l_leg_kny_pub, l_leg_uay_pub, r_leg_lhy_pub, r_leg_kny_pub, r_leg_uay_pub

    #global bool_time
    #bool_time = True

    # Initialize publishers
    pub_initial()
    global i_error, last_time
    i_error = 0
    last_time = 0

    global curr_time
    curr_time = 0

    global cur_2, cur_3, cur_4
    global Rcur_2, Rcur_3, Rcur_4
    cur_2 = 0.0
    cur_3 = 0.0
    cur_4 = 0.0
    Rcur_2 = 0.0
    Rcur_3 = 0.0
    Rcur_4 = 0.0



    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/gazebo/link_states", LinkStates, callback)
    rospy.Subscriber("/atlas_bipedal/joint_states", JointState, callback_torque)
                     # topic
    


    rospy.loginfo("Yes this is my python!!!!!!!!!")




    #l_leg_lhy_pub = rospy.Publisher('atlas_bipedal/l_leg_lhy_controller/command', Float64, queue_size=10, latch = True)
    #l_leg_lhy_pub.publish(0.5)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
if __name__ == '__main__':
    listener()

