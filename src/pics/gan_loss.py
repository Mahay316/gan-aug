import matplotlib.pyplot as plt
import numpy as np

from pic_utils import set_axes

dis_real_loss = []
dis_fake_loss = []
gen_loss = []
with open('./train.txt', 'r') as f:
    for line in f:
        result = line.split(', ')
        dis_real_loss.append(float(result[0]))
        dis_fake_loss.append(float(result[1]) * 30)
        gen_loss.append(float(result[2]))

steps_per_epoch = 215
training_step = np.arange(steps_per_epoch, (len(dis_fake_loss) + 1) * steps_per_epoch, step=steps_per_epoch)

fig, ax = plt.subplots(dpi=300)

ax.plot(training_step, dis_real_loss, label='critic-real')
ax.plot(training_step,dis_fake_loss, label=r'critic-fake$\times 30$')
ax.plot(training_step,gen_loss, label='gen')

ax.set_ylabel('Cross Entropy Loss')
ax.set_xlabel('Training Steps')
ax.set_ylim(0, 6)
ax.set_xlim(0)
ax.legend(loc='best')
ax.grid(linestyle=':')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.show()
