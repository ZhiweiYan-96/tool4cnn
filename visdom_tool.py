from visdom import Visdom 
import numpy as np 

def create_environment(env_name,port=8097,host_name="http://localhost"):
	print('Creating environment....')
	return Visdom(env=env_name,port=port,server=host_name)

def create_one_var_plot(env,_xlabel,_ylabel,_title,_legend):
	return env.line(
		X = np.zeros((1,)),
		Y = np.zeros(1,1),
		opts = dict(
			xlabel = _xlabel,
			ylabel = _ylabel,
			title = _title,
			legend = _legend
		)
	)

def update_one_var_plot(env,x_vec,y_vec,window,update_type):
	env.line(
		X = x_vec,
		Y = y_vec.unsqueze(0),
		win= window,
		update = update_type
	)

def create_n_variables_plot(env,_xlabel,_ylabels,_title,_legend,X,Y):
	#print('len legend:'+str(len(_legend)))
	return env.line(
		X = np.ones((1,len(Y)))*X,
		Y = np.expand_dims(Y,0),
		opts = dict(
			xlabel = _xlabel,
			ylabel = _ylabels,
			title = _title,
			legend = _legend
		)
	)



def update_n_variables_plot(env,x_vec,y_vec,window,update_type):
	#print("x_vec",x_vec)
	# print(x_vec)
	# print(y_vec)
	# print(np.expand_dims(y_vec,0))
	# print(np.ones((1, len(y_vec))) * x_vec)
	env.line(
		X = np.ones((1,len(y_vec)))*x_vec,
		Y = np.expand_dims(y_vec,0),
		win = window,
		update = update_type

	)

def create_heat_plot(env,width,height):
	return env.heatmap(
		X = np.random.randn(width,height),
		opts = dict(
			colormap='Electric'
		)

	)

def update_heat_plot(env,X,window,update_type):
	if update_type=="replace":
		env.heatmap(
			X =  np.flipud(X),
			win= window,
			# update= update_type,
		)
		return [window]
	elif update_type=="append":
		new_window = env.heatmap(
			X = X
		)
		return [window,new_window]

def create_img_plot(env,width,height,title,caption):
	return env.image(
		np.random.randn(3,width,height),
		opts = dict( 
			title = title,
			caption = caption
		)
	)

def update_img_plot(env,img,window,update_type):
	if update_type == "replace":
		env.image(
			img,
			win = window
		)
		return [window]
	elif update_type == "append":
		new_window= env.image(
			img
		)
		return [window,new_window]

# def create_hist_plot(env,)

def normalize(X):
	return (X-np.min(X))/(np.max(X)-np.min(X))*255

def create_hist_plot(env,size,title):
	return env.histogram(
		np.random.randn(1,size),
		opts = dict(
			numbins = 256,
			title = title,
		)
	)

def update_hist_plot(env,X_1d,window,update_type,title,numbins=30):
	if update_type=="replace":
		return [  env.histogram(X_1d,win=window,opts=dict(numbins=numbins,title=title)) ]
	elif update_type=="append":
		return [window,env.histogram(X_1d,opts = dict(numbins=numbins,title=title))]

if __name__ =='__main__':
	import numpy as np
	env = create_environment('test')
	X = np.array([1,2,3])
	Y = np.array([4,5,6])
	window= create_n_variables_plot(env,'x','y','test',['x1'],X,Y)
	# print('here')
	# heat = create_heat_plot(env,38,38)
	# new_X  = np.random.randn(38,38)
	# hist_plot = create_hist_plot(env,50000,'hist')
	# import time
	# time.sleep(2)
	# # update_heat_plot(env,new_X,heat,'append')
	# # update_heat_plot(env,new_X,heat,'append')
	# a = np.random.randn(1,5000)
	# update_hist_plot(env,a,100,hist_plot,'replace','hist1')
	#
	# #env.histogram(X=normalize(new_X.flatten()),opts=dict(numbins=256)
	# # hist = create_hist_plot(env,38)
	# time.sleep(2)
	# a = np.random.randn(1,8000)
	# update_hist_plot(env,a,200,hist_plot,'replace','hist2')


	# update_hist_plot(env,new_X.flatten(),hist,'replace')
	# update_hist_plot(env,new_X.flatten(),hist,'append')

