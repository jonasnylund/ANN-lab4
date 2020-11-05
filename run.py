from util import *
from rbm import RestrictedBoltzmannMachine 
from dbn import DeepBeliefNet
from matplotlib import pyplot as plt

if __name__ == "__main__":

    image_size = [28,28]
    train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)
    print(train_imgs.shape)
    
    ''' restricted boltzmann machine '''
    
    print ("\nStarting a Restricted Boltzmann Machine..")
    
    # rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
    #                                   ndim_hidden=200,
    #                                   is_bottom=True,
    #                                   image_size=image_size,
    #                                   is_top=False,
    #                                   n_labels=10,
    #                                   batch_size=20
    #  )
    
    # rbm.cd1(visible_trainset=train_imgs[:,:], n_iterations=10000)
    
    # K = 1

    # for iii in range(5):
    
	   #  img = test_imgs[20+iii,:][np.newaxis,:]
	   #  # print(img.shape)
	   #  h_n = rbm.get_h_given_v(img)[1]
	    
	   #  for k in range(K):
	   #      v_n, _ = rbm.get_v_given_h(h_n)
	   #      _, h_n = rbm.get_h_given_v(v_n)
	    
	   #  out = rbm.get_v_given_h(h_n)[0]
	    
	   #  plt.subplot(2,5,iii+1)
	   #  plt.imshow(img.reshape((28,28)))
	   #  plt.subplot(2,5,iii+5+1)
	   #  plt.imshow(out.reshape((28,28)))
        
    # plt.savefig("testimg_rbm")


    ''' deep- belief net '''

    print ("\nStarting a Deep Belief Net..")
    
    dbn = DeepBeliefNet(sizes={"vis":image_size[0]*image_size[1], "hid":500, "pen":500, "top":2000, "lbl":10},
                        image_size=image_size,
                        n_labels=10,
                        batch_size=10
    )
    
    ''' greedy layer-wise training '''

    dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=10000)

    # for i in range(25):
    # 	ax = plt.subplot(5,5,i+1);
    # 	plt.imshow(test_imgs[i, :].reshape(image_size));
    # 	plt.title("t: {} p: {}".format(np.argmax(test_lbls[i,:]), np.argmax(dbn.recognize(test_imgs[i, :][np.newaxis,:], test_lbls[i,:][np.newaxis,:]))))
    # plt.tight_layout();
    # plt.show()

    # dbn.recognize(train_imgs[:10000], train_lbls[:10000])
    # dbn.recognize(test_imgs, test_lbls)

    for digit in range(10):
        digit_1hot = np.zeros(shape=(1,10))
        digit_1hot[0,digit] = 1
        dbn.generate(digit_1hot, name="rbms")

    ''' fine-tune wake-sleep training '''

    # dbn.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=10000)

    # dbn.recognize(train_imgs, train_lbls)
    
    # dbn.recognize(test_imgs, test_lbls)
    
    # for digit in range(10):
    #     digit_1hot = np.zeros(shape=(1,10))
    #     digit_1hot[0,digit] = 1
    #     dbn.generate(digit_1hot, name="dbn")
