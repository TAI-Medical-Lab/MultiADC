import re
import numpy as np
se_list = []
sp_list = []
mcc_list = []
acc_list = []
auc_list = []
F1_list = []
BA_list = []
prauc_list = []
PPV_list = []
NPV_list = []

# 日志文本
# log_line = '''1210 2035
# 12-10 20:36:48 epoch:53 test_loss0.06658823879157331 se:0.9245 sp:0.8235 mcc:0.7569 acc:0.8851 auc:0.9467 F1:0.9074 BA:0.874 prauc:0.9656 PPV:0.8909 NPV:0.875
# 12-10 20:37:50 epoch:28 test_loss0.11172678903944191 se:0.8571 sp:0.8065 mcc:0.655 acc:0.8391 auc:0.894 F1:0.8727 BA:0.8318 prauc:0.9044 PPV:0.8889 NPV:0.7576
# 12-10 20:39:31 epoch:39 test_loss0.12606761613111386 se:0.8727 sp:0.7812 mcc:0.654 acc:0.8391 auc:0.8943 F1:0.8727 BA:0.827 prauc:0.8796 PPV:0.8727 NPV:0.7812
# 12-10 20:42:24 epoch:89 test_loss0.2981957012209399 se:0.8621 sp:0.6207 mcc:0.4972 acc:0.7816 auc:0.7818 F1:0.8403 BA:0.7414 prauc:0.8833 PPV:0.8197 NPV:0.6923
# 12-10 20:43:36 epoch:35 test_loss0.12695559391448663 se:0.8793 sp:0.75 mcc:0.6293 acc:0.8372 auc:0.8849 F1:0.8793 BA:0.8147 prauc:0.9379 PPV:0.8793 NPV:0.75
# '''

#只有虚拟图
# log_line = '''1210 2139

# 2348 12-10 23:51:55 epoch:29 test_loss0.06753424816261763 se:0.8571 sp:0.7419 mcc:0.5991 acc:0.8161 auc:0.8894 F1:0.8571 BA:0.7995 prauc:0.9305 PPV:0.8571 NPV:0.7419
# 12-10 21:42:42 epoch:60 test_loss0.09785004893596146 se:0.8214 sp:0.871 mcc:0.6707 acc:0.8391 auc:0.9101 F1:0.8679 BA:0.8462 prauc:0.9421 PPV:0.92 NPV:0.7297
# 12-10 21:45:38 epoch:68 test_loss0.07151520192280583 se:0.7636 sp:0.9062 mcc:0.6464 acc:0.8161 auc:0.9165 F1:0.84 BA:0.8349 prauc:0.9522 PPV:0.9333 NPV:0.6905
# 12-10 21:46:19 epoch:18 test_loss0.06940247675125626 se:0.8621 sp:0.5862 mcc:0.467 acc:0.7701 auc:0.7782 F1:0.8333 BA:0.7241 prauc:0.8727 PPV:0.8065 NPV:0.68
# 12-10 21:47:52 epoch:19 test_loss0.05576055999412093 se:0.8966 sp:0.7143 mcc:0.6232 acc:0.8372 auc:0.891 F1:0.8814 BA:0.8054 prauc:0.9429 PPV:0.8667 NPV:0.7692
# '''

# #只有指纹
# log_line='''
# 12-11 00:04:19 epoch:33 test_loss0.09719153168215149 se:0.9434 sp:0.7941 mcc:0.7571 acc:0.8851 auc:0.934 F1:0.9091 BA:0.8688 prauc:0.9556 PPV:0.8772 NPV:0.9
# 12-11 00:05:37 epoch:48 test_loss0.20729798027153673 se:0.8929 sp:0.7742 mcc:0.6721 acc:0.8506 auc:0.8888 F1:0.885 BA:0.8335 prauc:0.9142 PPV:0.8772 NPV:0.8
# 12-11 00:05:38 epoch:49 test_loss0.20496389663767542 se:0.875 sp:0.7742 mcc:0.6492 acc:0.8391 auc:0.8894 F1:0.875 BA:0.8246 prauc:0.9178 PPV:0.875 NPV:0.7742
# 12-11 00:07:53 epoch:56 test_loss0.30318856547618733 se:0.7931 sp:0.6552 mcc:0.4412 acc:0.7471 auc:0.7925 F1:0.807 BA:0.7241 prauc:0.8874 PPV:0.8214 NPV:0.6129
# 12-11 00:08:41 epoch:31 test_loss0.18184965883576593 se:0.8793 sp:0.7143 mcc:0.5993 acc:0.8256 auc:0.8824 F1:0.8718 BA:0.7968 prauc:0.933 PPV:0.8644 NPV:0.7407
# '''


# 没有pca降维 1211-2022
log_line='''
12-11 20:25:03 epoch:87 test_loss0.08118921809497921 se:0.8302 sp:0.8529 mcc:0.672 acc:0.8391 auc:0.8774 F1:0.8627 BA:0.8416 prauc:0.9164 PPV:0.898 NPV:0.7632
12-11 20:27:23 epoch:71 test_loss0.12306040945066803 se:0.8393 sp:0.7742 mcc:0.6055 acc:0.8161 auc:0.8813 F1:0.8545 BA:0.8067 prauc:0.8814 PPV:0.8704 NPV:0.7273
12-11 20:29:05 epoch:49 test_loss0.0704233346228627 se:0.6727 sp:0.875 mcc:0.5291 acc:0.7471 auc:0.8909 F1:0.7708 BA:0.7739 prauc:0.9252 PPV:0.9024 NPV:0.6087
12-11 20:31:09 epoch:67 test_loss0.22137806247705702 se:0.8103 sp:0.6207 mcc:0.431 acc:0.7471 auc:0.7568 F1:0.8103 BA:0.7155 prauc:0.8398 PPV:0.8103 NPV:0.6207
12-11 20:34:10 epoch:86 test_loss0.1046389618585276 se:0.9138 sp:0.6786 mcc:0.6188 acc:0.8372 auc:0.8959 F1:0.8833 BA:0.7962 prauc:0.949 PPV:0.8548 NPV:0.7917
# '''

# #sum
# log_line='''
# 第1折---se:0.9245 sp:0.7941 mcc:0.7322 acc:0.8736 auc:0.944 F1:0.8991 BA:0.8593 prauc:0.9635 PPV:0.875 NPV:0.871
# 第2折---se:0.8393 sp:0.7742 mcc:0.6055 acc:0.8161 auc:0.8894 F1:0.8545 BA:0.8067 prauc:0.9249 PPV:0.8704 NPV:0.7273
# 第3折---se:0.8727 sp:0.7812 mcc:0.654 acc:0.8391 auc:0.8773 F1:0.8727 BA:0.827 prauc:0.8877 PPV:0.8727 NPV:0.7812
# 第4折---se:0.7759 sp:0.6552 mcc:0.4214 acc:0.7356 auc:0.7812 F1:0.7965 BA:0.7155 prauc:0.8874 PPV:0.8182 NPV:0.5938
# 12-11 22:28:16 epoch:46 test_loss0.13111817594184433 se:0.8966 sp:0.75 mcc:0.6528 acc:0.8488 auc:0.8799 F1:0.8889 BA:0.8233 prauc:0.9196 PPV:0.8814 NPV:0.7778
# '''

#max
# log_line='''
# 12-11 22:22:30 epoch:54 test_loss0.0636955034630052 se:0.9057 sp:0.8235 mcc:0.7333 acc:0.8736 auc:0.9417 F1:0.8972 BA:0.8646 prauc:0.9634 PPV:0.8889 NPV:0.8485
# 第2折---se:0.875 sp:0.7742 mcc:0.6492 acc:0.8391 auc:0.8911 F1:0.875 BA:0.8246 prauc:0.9025 PPV:0.875 NPV:0.7742
# 第3折---se:0.8727 sp:0.75 mcc:0.627 acc:0.8276 auc:0.8915 F1:0.8649 BA:0.8114 prauc:0.9067 PPV:0.8571 NPV:0.7742
# 第4折---se:0.8276 sp:0.6207 mcc:0.4523 acc:0.7586 auc:0.7652 F1:0.8205 BA:0.7241 prauc:0.8544 PPV:0.8136 NPV:0.6429
# 12-11 22:32:31 epoch:62 test_loss0.157675972511602 se:0.9138 sp:0.7143 mcc:0.6481 acc:0.8488 auc:0.8929 F1:0.8908 BA:0.814 prauc:0.9438 PPV:0.8689 NPV:0.8
# '''

#mean
# log_line='''
# 第1折---se:0.9057 sp:0.8529 mcc:0.7586 acc:0.8851 auc:0.9456 F1:0.9057 BA:0.8793 prauc:0.9658 PPV:0.9057 NPV:0.8529
# 第2折---se:0.8393 sp:0.7742 mcc:0.6055 acc:0.8161 auc:0.8911 F1:0.8545 BA:0.8067 prauc:0.89 PPV:0.8704 NPV:0.7273
# 12-11 22:31:34 epoch:60 test_loss0.12315055462478221 se:0.8727 sp:0.7812 mcc:0.654 acc:0.8391 auc:0.8989 F1:0.8727 BA:0.827 prauc:0.9267 PPV:0.8727 NPV:0.7812
# 12-11 22:34:00 epoch:71 test_loss0.28209022509640663 se:0.8276 sp:0.6552 mcc:0.4828 acc:0.7701 auc:0.7622 F1:0.8276 BA:0.7414 prauc:0.8535 PPV:0.8276 NPV:0.6552
# 12-11 22:35:20 epoch:38 test_loss0.13266099538913992 se:0.9138 sp:0.7143 mcc:0.6481 acc:0.8488 auc:0.883 F1:0.8908 BA:0.814 prauc:0.9316 PPV:0.8689 NPV:0.8
# '''

####################3new



#playload new 1536
# log_line='''
# 128 0.00009 criterion = BCEFocalLoss(gamma=3, alpha=0.25)
# 12-16 15:37:18 epoch:36 test_loss0.03036057876377571 se:0.6441 sp:0.6957 mcc:0.3067 acc:0.6585 auc:0.7424 F1:0.7308 BA:0.6699 prauc:0.8868 PPV:0.8444 NPV:0.4324

# 12-16 15:53:14 epoch:21 test_loss0.036400847136974335 se:0.3953 sp:1.0 mcc:0.487 acc:0.6829 auc:0.811 F1:0.5667 BA:0.6977 prauc:0.8487 PPV:1.0 NPV:0.6

# 12-16 15:39:19 epoch:35 test_loss0.05105644046533399 se:0.6154 sp:0.7907 mcc:0.4135 acc:0.7073 auc:0.7901 F1:0.6667 BA:0.703 prauc:0.7737 PPV:0.7273 NPV:0.6939

# 12-16 15:40:28 epoch:39 test_loss0.021623839624226093 se:0.7167 sp:0.9091 mcc:0.5572 acc:0.7683 auc:0.8917 F1:0.819 BA:0.8129 prauc:0.9628 PPV:0.9556 NPV:0.5405

# 12-16 15:41:22 epoch:25 test_loss0.03312840503526897 se:0.7353 sp:0.7143 mcc:0.3567 acc:0.7317 auc:0.6775 F1:0.8197 BA:0.7248 prauc:0.8728 PPV:0.9259 NPV:0.3571

# '''

#PLAYLOAD 1216 2217 final
# log_line='''
# 12-16 22:18:50 epoch:24 test_loss0.031796994187482976 se:0.678 sp:0.6957 mcc:0.3393 acc:0.6829 auc:0.6931 F1:0.7547 BA:0.6868 prauc:0.8786 PPV:0.8511 NPV:0.4571

# 12-16 22:20:16 epoch:31 test_loss0.03392822272712138 se:0.4884 sp:0.9487 mcc:0.4859 acc:0.7073 auc:0.7901 F1:0.6364 BA:0.7185 prauc:0.848 PPV:0.913 NPV:0.6271

# 12-16 22:21:29 epoch:18 test_loss0.05326281033638047 se:0.6923 sp:0.7442 mcc:0.4371 acc:0.7195 auc:0.7847 F1:0.7013 BA:0.7182 prauc:0.7548 PPV:0.7105 NPV:0.7273

# 12-16 22:23:16 epoch:19 test_loss0.02184908594027525 se:0.6667 sp:0.9091 mcc:0.5104 acc:0.7317 auc:0.8879 F1:0.7843 BA:0.7879 prauc:0.9584 PPV:0.9524 NPV:0.5

# 12-16 22:24:49 epoch:38 test_loss0.04829793332553491 se:0.8529 sp:0.5714 mcc:0.3858 acc:0.8049 auc:0.688 F1:0.8788 BA:0.7122 prauc:0.8742 PPV:0.9062 NPV:0.4444

# '''
# # antibody cold  1216 1345 final
log_line='''
12-16 13:46:03 epoch:3 test_loss0.038779212811658546 se:0.0794 sp:0.9565 mcc:0.0624 acc:0.314 auc:0.4023 F1:0.1449 BA:0.5179 prauc:0.7074 PPV:0.8333 NPV:0.275
12-16 13:49:02 epoch:14 test_loss0.020452751366551534 se:0.9865 sp:0.5833 mcc:0.6798 acc:0.9302 auc:0.8255 F1:0.9605 BA:0.7849 prauc:0.9639 PPV:0.9359 NPV:0.875
12-16 13:53:53 epoch:78 test_loss0.03838372083250866 se:0.5 sp:0.8594 mcc:0.3712 acc:0.7674 auc:0.7553 F1:0.5238 BA:0.6797 prauc:0.6437 PPV:0.55 NPV:0.8333
12-16 13:55:49 epoch:59 test_loss0.04923437778339829 se:0.8393 sp:0.5667 mcc:0.4213 acc:0.7442 auc:0.7649 F1:0.8103 BA:0.703 prauc:0.8274 PPV:0.7833 NPV:0.6538

12-16 13:57:29 epoch:45 test_loss0.05737118981778622 se:0.7188 sp:0.8182 mcc:0.4749 acc:0.7442 auc:0.7045 F1:0.807 BA:0.7685 prauc:0.8569 PPV:0.92 NPV:0.5



'''
log_line='''
12-16 13:46:03 epoch:3 test_loss0.038779212811658546 se:0.0794 sp:0.9565 mcc:0.0624 acc:0.314 auc:0.4023 F1:0.1449 BA:0.5179 prauc:0.7074 PPV:0.8333 NPV:0.275
12-16 13:50:28 epoch:70 test_loss0.028933774593264558 se:0.9865 sp:0.25 mcc:0.3891 acc:0.8837 auc:0.777 F1:0.9359 BA:0.6182 prauc:0.9294 PPV:0.8902 NPV:0.75
12-16 13:53:55 epoch:79 test_loss0.039074801515008126 se:0.5 sp:0.8594 mcc:0.3712 acc:0.7674 auc:0.756 F1:0.5238 BA:0.6797 prauc:0.6438 PPV:0.55 NPV:0.8333
12-16 13:55:49 epoch:59 test_loss0.04923437778339829 se:0.8393 sp:0.5667 mcc:0.4213 acc:0.7442 auc:0.7649 F1:0.8103 BA:0.703 prauc:0.8274 PPV:0.7833 NPV:0.6538

12-16 13:56:56 epoch:26 test_loss0.04881938386621863 se:0.6562 sp:0.8636 mcc:0.4542 acc:0.7093 auc:0.7259 F1:0.7706 BA:0.7599 prauc:0.8629 PPV:0.9333 NPV:0.4634



'''


#1216 1918 our model 6452 增加了上采样增加了val
# log_line='''
# 12-16 19:19:34 epoch:34 test_loss0.07373042765405328 se:0.9245 sp:0.8235 mcc:0.7569 acc:0.8851 auc:0.9234 F1:0.9074 BA:0.874 prauc:0.9476 PPV:0.8909 NPV:0.875
# 12-16 19:20:51 epoch:29 test_loss0.26230949334714604 se:0.8571 sp:0.6774 mcc:0.5431 acc:0.7931 auc:0.8278 F1:0.8421 BA:0.7673 prauc:0.8221 PPV:0.8276 NPV:0.7241
# 12-16 19:21:57 epoch:28 test_loss0.10770568144561231 se:0.8182 sp:0.8438 mcc:0.6456 acc:0.8276 auc:0.8903 F1:0.8571 BA:0.831 prauc:0.8969 PPV:0.9 NPV:0.7297
# 12-16 19:23:53 epoch:49 test_loss0.13012806745781297 se:0.8276 sp:0.7931 mcc:0.603 acc:0.8161 auc:0.8543 F1:0.8571 BA:0.8103 prauc:0.9149 PPV:0.8889 NPV:0.697
# 12-16 19:25:09 epoch:38 test_loss0.08014100686062214 se:0.9138 sp:0.75 mcc:0.6773 acc:0.8605 auc:0.9046 F1:0.8983 BA:0.8319 prauc:0.9533 PPV:0.8833 NPV:0.8077
#  '''


# log_line='''
# 12-16 19:19:34 epoch:34 test_loss0.07373042765405328 se:0.9245 sp:0.8235 mcc:0.7569 acc:0.8851 auc:0.9234 F1:0.9074 BA:0.874 prauc:0.9476 PPV:0.8909 NPV:0.875
# 12-16 19:20:51 epoch:29 test_loss0.26230949334714604 se:0.8571 sp:0.6774 mcc:0.5431 acc:0.7931 auc:0.8278 F1:0.8421 BA:0.7673 prauc:0.8221 PPV:0.8276 NPV:0.7241
# 12-16 19:21:57 epoch:28 test_loss0.10770568144561231 se:0.8182 sp:0.8438 mcc:0.6456 acc:0.8276 auc:0.8903 F1:0.8571 BA:0.831 prauc:0.8969 PPV:0.9 NPV:0.7297
# 12-16 19:23:53 epoch:49 test_loss0.13012806745781297 se:0.8276 sp:0.7931 mcc:0.603 acc:0.8161 auc:0.8543 F1:0.8571 BA:0.8103 prauc:0.9149 PPV:0.8889 NPV:0.697
# 12-16 19:25:09 epoch:38 test_loss0.08014100686062214 se:0.9138 sp:0.75 mcc:0.6773 acc:0.8605 auc:0.9046 F1:0.8983 BA:0.8319 prauc:0.9533 PPV:0.8833 NPV:0.8077
#  '''

#64 0.00005 focal 0.4
# mean 1216 2301 cat 
# log_line='''
# 12-16 23:02:13 epoch:31 test_loss0.04576134709534289 se:0.8491 sp:0.9118 mcc:0.7465 acc:0.8736 auc:0.9245 F1:0.8911 BA:0.8804 prauc:0.947 PPV:0.9375 NPV:0.7949
# 12-16 23:03:34 epoch:41 test_loss0.1478425846702751 se:0.8393 sp:0.7097 mcc:0.549 acc:0.7931 auc:0.8381 F1:0.8393 BA:0.7745 prauc:0.8526 PPV:0.8393 NPV:0.7097
# 12-16 23:04:36 epoch:29 test_loss0.068550079825452 se:0.7091 sp:0.9062 mcc:0.5938 acc:0.7816 auc:0.9023 F1:0.8041 BA:0.8077 prauc:0.8983 PPV:0.9286 NPV:0.6444
# 12-16 23:06:21 epoch:51 test_loss0.061705302061705755 se:0.7759 sp:0.7586 mcc:0.5138 acc:0.7701 auc:0.8674 F1:0.8182 BA:0.7672 prauc:0.9226 PPV:0.8654 NPV:0.6286
# 12-16 23:08:02 epoch:45 test_loss0.05198311268590217 se:0.8448 sp:0.8571 mcc:0.6764 acc:0.8488 auc:0.907 F1:0.8829 BA:0.851 prauc:0.9545 PPV:0.9245 NPV:0.7273
#  '''

# sum 1216 2309 cat 
# log_line='''
# 12-16 23:11:55 epoch:68 test_loss0.04713452112828863 se:0.9057 sp:0.8824 mcc:0.7841 acc:0.8966 auc:0.9218 F1:0.9143 BA:0.894 prauc:0.9488 PPV:0.9231 NPV:0.8571
# 12-16 23:13:19 epoch:39 test_loss0.11501007758337876 se:0.8393 sp:0.8065 mcc:0.6338 acc:0.8276 auc:0.8474 F1:0.8624 BA:0.8229 prauc:0.8374 PPV:0.8868 NPV:0.7353
# 12-16 23:15:04 epoch:48 test_loss0.0794843318711581 se:0.7091 sp:0.9375 mcc:0.6246 acc:0.7931 auc:0.9 F1:0.8125 BA:0.8233 prauc:0.9253 PPV:0.9512 NPV:0.6522
# 12-16 23:17:10 epoch:58 test_loss0.07505776382040703 se:0.7241 sp:0.7931 mcc:0.4903 acc:0.7471 auc:0.8478 F1:0.7925 BA:0.7586 prauc:0.9155 PPV:0.875 NPV:0.5897
# 12-16 23:18:38 epoch:40 test_loss0.0544190488235895 se:0.8103 sp:0.8214 mcc:0.6055 acc:0.814 auc:0.8996 F1:0.8545 BA:0.8159 prauc:0.9469 PPV:0.9038 NPV:0.6765
#  '''
# # max 1216 2356 cat 
# log_line='''
# 12-16 23:57:09 epoch:31 test_loss0.04637247767170955 se:0.8491 sp:0.9118 mcc:0.7465 acc:0.8736 auc:0.924 F1:0.8911 BA:0.8804 prauc:0.9467 PPV:0.9375 NPV:0.7949
# 12-16 23:58:29 epoch:38 test_loss0.1503774206871274 se:0.8393 sp:0.7097 mcc:0.549 acc:0.7931 auc:0.8318 F1:0.8393 BA:0.7745 prauc:0.8451 PPV:0.8393 NPV:0.7097
# 12-17 00:01:07 epoch:75 test_loss0.07398252224485422 se:0.7818 sp:0.875 mcc:0.6355 acc:0.8161 auc:0.9062 F1:0.8431 BA:0.8284 prauc:0.8859 PPV:0.9149 NPV:0.7
# 12-17 00:03:18 epoch:68 test_loss0.06912374393693332 se:0.7414 sp:0.7586 mcc:0.4768 acc:0.7471 auc:0.8567 F1:0.7963 BA:0.75 prauc:0.9222 PPV:0.86 NPV:0.5946
# 12-17 00:04:27 epoch:30 test_loss0.05170160887199779 se:0.8448 sp:0.8214 mcc:0.6459 acc:0.8372 auc:0.9009 F1:0.875 BA:0.8331 prauc:0.9512 PPV:0.9074 NPV:0.7188
#  '''

# 只有Morgan
# 1217 00006
# log_line='''
# 第1折---se:0.9245 sp:0.8824 mcc:0.8069 acc:0.908 auc:0.9223 F1:0.9245 BA:0.9034 prauc:0.9482 PPV:0.9245 NPV:0.8824
# 12-17 00:07:46 epoch:47 test_loss0.16316681004118647 se:0.8571 sp:0.7097 mcc:0.5711 acc:0.8046 auc:0.8278 F1:0.8496 BA:0.7834 prauc:0.8188 PPV:0.8421 NPV:0.7333
# 12-17 00:09:40 epoch:57 test_loss0.0802780182218586 se:0.7636 sp:0.875 mcc:0.6169 acc:0.8046 auc:0.8994 F1:0.8317 BA:0.8193 prauc:0.8829 PPV:0.913 NPV:0.6829
# 12-17 00:10:56 epoch:34 test_loss0.06373725882891951 se:0.7069 sp:0.7931 mcc:0.4729 acc:0.7356 auc:0.8591 F1:0.781 BA:0.75 prauc:0.9269 PPV:0.8723 NPV:0.575
# 12-17 00:12:44 epoch:50 test_loss0.051560449305661886 se:0.8621 sp:0.8214 mcc:0.6671 acc:0.8488 auc:0.9107 F1:0.885 BA:0.8417 prauc:0.9592 PPV:0.9091 NPV:0.7419
#  '''

# 没有PCA
# 1217 1704
# log_line='''
# 12-17 17:06:14 epoch:44 test_loss0.0654192663643552 se:0.9434 sp:0.5588 mcc:0.5638 acc:0.7931 auc:0.8196 F1:0.8475 BA:0.7511 prauc:0.8656 PPV:0.7692 NPV:0.8636
# 12-17 17:09:45 epoch:105 test_loss0.14211909481506238 se:0.875 sp:0.7097 mcc:0.594 acc:0.8161 auc:0.8358 F1:0.8596 BA:0.7923 prauc:0.8384 PPV:0.8448 NPV:0.7586
# 12-17 17:11:14 epoch:43 test_loss0.055415889655036486 se:0.7273 sp:0.8125 mcc:0.5214 acc:0.7586 auc:0.8665 F1:0.7921 BA:0.7699 prauc:0.9171 PPV:0.8696 NPV:0.6341
# 12-17 17:13:05 epoch:55 test_loss0.11298106405241735 se:0.7586 sp:0.5517 mcc:0.3078 acc:0.6897 auc:0.7093 F1:0.7652 BA:0.6552 prauc:0.8384 PPV:0.7719 NPV:0.5333
# 12-17 17:15:49 epoch:89 test_loss0.05702650061873502 se:0.8276 sp:0.8571 mcc:0.6563 acc:0.8372 auc:0.9126 F1:0.8727 BA:0.8424 prauc:0.9577 PPV:0.9231 NPV:0.7059
#  '''


# antigen cold  1218 0114 final
# batch_size:64
# lr:0.00008
# alpha=0.25,gamma=3，删除了val
# log_line='''
# 12-18 01:15:21 epoch:6 test_loss0.10581645185564771 se:0.9259 sp:0.037 mcc:-0.0725 acc:0.6296 auc:0.404 F1:0.7692 BA:0.4815 prauc:0.5804 PPV:0.6579 NPV:0.2
# 12-18 01:17:00 epoch:21 test_loss0.014337400225979954 se:0.8293 sp:0.95 mcc:0.7841 acc:0.8889 auc:0.953 F1:0.8831 BA:0.8896 prauc:0.9646 PPV:0.9444 NPV:0.8444
# 12-18 01:17:28 epoch:4 test_loss0.03733033176373552 se:0.8108 sp:0.7273 mcc:0.5364 acc:0.7654 auc:0.844 F1:0.7595 BA:0.769 prauc:0.8202 PPV:0.7143 NPV:0.8205
# 12-18 01:18:49 epoch:16 test_loss0.15219619870185852 se:0.9831 sp:0.0455 mcc:0.0817 acc:0.7284 auc:0.5743 F1:0.8406 BA:0.5143 prauc:0.7771 PPV:0.7342 NPV:0.5
# 12-18 01:20:53 epoch:48 test_loss0.0300015750122659 se:0.6056 sp:0.8 mcc:0.2685 acc:0.6296 auc:0.6408 F1:0.7414 BA:0.7028 prauc:0.9385 PPV:0.9556 NPV:0.2222
# '''




# 0046之后为只有adc，虚拟节点 取消val，取消学习率weight decay alpha 0.3
# batch_size:128
# lr:0.0002
# #########1217 1342
# log_line='''
# 12-17 13:43:16 epoch:26 test_loss0.050839895275474965 se:0.8868 sp:0.6765 mcc:0.583 acc:0.8046 auc:0.8618 F1:0.8468 BA:0.7816 prauc:0.8999 PPV:0.8103 NPV:0.7931
# 12-17 13:45:33 epoch:51 test_loss0.0885464022385663 se:0.75 sp:0.871 mcc:0.5958 acc:0.7931 auc:0.8831 F1:0.8235 BA:0.8105 prauc:0.9131 PPV:0.913 NPV:0.6585
# 12-17 13:46:15 epoch:17 test_loss0.05433882400393486 se:0.9818 sp:0.5938 mcc:0.6596 acc:0.8391 auc:0.8364 F1:0.8852 BA:0.7878 prauc:0.8339 PPV:0.806 NPV:0.95
# 12-17 13:47:58 epoch:19 test_loss0.06919310909920726 se:0.7586 sp:0.5862 mcc:0.3394 acc:0.7011 auc:0.7521 F1:0.7719 BA:0.6724 prauc:0.8283 PPV:0.7857 NPV:0.5484
# 12-17 13:50:01 epoch:54 test_loss0.13534029382605886 se:0.8793 sp:0.75 mcc:0.6293 acc:0.8372 auc:0.8744 F1:0.8793 BA:0.8147 prauc:0.9267 PPV:0.8793 NPV:0.75
#  '''
# log_line='''1217 1621 final 只有虚拟图
# 12-17 16:24:34 epoch:89 test_loss0.1042904481291771 se:0.7736 sp:0.9412 mcc:0.6976 acc:0.8391 auc:0.9007 F1:0.8542 BA:0.8574 prauc:0.9322 PPV:0.9535 NPV:0.7273
# 12-17 16:26:21 epoch:48 test_loss0.08122668415307999 se:0.8036 sp:0.7742 mcc:0.5643 acc:0.7931 auc:0.8727 F1:0.8333 BA:0.7889 prauc:0.916 PPV:0.8654 NPV:0.6857
# 12-17 16:32:15 epoch:102 test_loss0.13082550466060638 se:0.8182 sp:0.875 mcc:0.6739 acc:0.8391 auc:0.8787 F1:0.8654 BA:0.8466 prauc:0.8976 PPV:0.9184 NPV:0.7368
# 12-17 16:34:09 epoch:26 test_loss0.07327618449926376 se:0.8103 sp:0.6552 mcc:0.4617 acc:0.7586 auc:0.7176 F1:0.8174 BA:0.7328 prauc:0.8079 PPV:0.8246 NPV:0.6333
# 12-17 16:36:43 epoch:60 test_loss0.12371651083230972 se:0.8276 sp:0.8214 mcc:0.6254 acc:0.8256 auc:0.8707 F1:0.8649 BA:0.8245 prauc:0.9276 PPV:0.9057 NPV:0.697
#  '''


# import torch
# feats = torch.rand(3, 4)
# print(feats)
# # 应用最大池化操作
# subgraph_max, _ = torch.max(feats, dim=0)  # 忽略索引
# print(subgraph_max)
# # 打印结果形状和部分值以确认正确性
# print("Max pooling result shape:", subgraph_max.shape)



pattern = r"se:(?P<se>\d+\.\d+)\s*sp:(?P<sp>\d+\.\d+)\s*mcc:(?P<mcc>-?\d+\.\d+)\s*acc:(?P<acc>\d+\.\d+)\s*auc:(?P<auc>\d+\.\d+)\s*F1:(?P<F1>\d+\.\d+)\s*BA:(?P<BA>\d+\.\d+)\s*prauc:(?P<prauc>\d+\.\d+)\s*PPV:(?P<PPV>\d+\.\d+)\s*NPV:(?P<NPV>\d+\.\d+)"
matches = re.findall(pattern, log_line)

for match in matches:
    # 提取匹配的数据并添加到对应的列表中
    se_list.append(float(match[0]))
    sp_list.append(float(match[1]))
    mcc_list.append(float(match[2]))
    acc_list.append(float(match[3]))
    auc_list.append(float(match[4]))
    F1_list.append(float(match[5]))
    BA_list.append(float(match[6]))
    prauc_list.append(float(match[7]))
    PPV_list.append(float(match[8]))
    NPV_list.append(float(match[9]))

# 计算均值和标准差
metrics = {
    'se': (np.mean(se_list), np.std(se_list)),
    'sp': (np.mean(sp_list), np.std(sp_list)),
    'mcc': (np.mean(mcc_list), np.std(mcc_list)),
    'acc': (np.mean(acc_list), np.std(acc_list)),
    'auc': (np.mean(auc_list), np.std(auc_list)),
    'F1': (np.mean(F1_list), np.std(F1_list)),
    'BA': (np.mean(BA_list), np.std(BA_list)),
    'prauc': (np.mean(prauc_list), np.std(prauc_list)),
    'PPV': (np.mean(PPV_list), np.std(PPV_list)),
    'NPV': (np.mean(NPV_list), np.std(NPV_list))
}

# 打印结果以验证
for metric, (mean, std) in metrics.items():
    print(f"{metric}_mean: {mean:.4f}, {metric}_std: {std:.4f}")
    with open("/HARD-DATA/YL/WY/ADC2/final.txt", 'a') as f:
        f.write(f"{metric}_mean: {mean:.4f}, {metric}_std: {std:.4f}")


  