import matplotlib.pyplot as pl
def cp(event): print 'asdfsdf'
fig=pl.figure()
pl.plot([1,2,3])
cid = fig.canvas.mpl_connect('button_press_event', cp)