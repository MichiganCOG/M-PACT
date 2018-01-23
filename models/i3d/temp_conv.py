import numpy as np

if __name__=="__main__":
    X = np.load('i3d_rgb_weights_tf_dim_ordering_tf_kernels.npy').tolist()
    for m in X.keys():
        if len(X[m].keys()) > 0:
            for n in X[m].keys():
                    if len(X[m][n].keys()) > 0:
                        for o in X[m][n].keys():
                                if len(X[m][n][o].keys()) > 0:
                                    for p in X[m][n][o].keys():
                                        try:
                                            if p == 'w' or p == 'b':
                                                print m,n,o
                                                if p == 'w':
                                                    X[m][n][o]['kernel'] = X[m][n][o].pop('w')
                                                else:
                                                    X[m][n][o]['bias'] = X[m][n][o].pop('b')
                                                break 
                                        except:
                                            #import pdb; pdb.set_trace()
                                            pass

                                        if len(X[m][n][o][p].keys()) > 0:
                                            for q in X[m][n][o][p].keys():
                                                try:
                                                    if q == 'w' or q == 'b':
                                                        print m,n,o,p
                                                        if q == 'w':
                                                            X[m][n][o][p]['kernel'] = X[m][n][o][p].pop('w')
                                                        else:
                                                            X[m][n][o][p]['bias'] = X[m][n][o][p].pop('b')
                                                        break 
                                                except:
                                                    #import pdb; pdb.set_trace()
                                                    pass

                                                try:
                                                    for r in X[m][n][o][p][q].keys():
                                                        try:
                                                            if r == 'w' or r == 'b':
                                                                print m,n,o,p,q
                                                                if r == 'w':
                                                                    X[m][n][o][p][q]['kernel'] = X[m][n][o][p][q].pop('w')
                                                                else:
                                                                    X[m][n][o][p][q]['bias'] = X[m][n][o][p][q].pop('b')
                                                                break 
                                                        except:
                                                            #import pdb; pdb.set_trace()
                                                            pass

                                                        try:
                                                            for s in X[m][n][o][p][q][r].keys():
                                                                try:
                                                                    if s == 'w' or s == 'b':
                                                                        print m,n,o,p,q,r
                                                                        if s == 'w':
                                                                            X[m][n][o][p][q][r]['kernel'] = X[m][n][o][p][q][r].pop('w')
                                                                        else:
                                                                            X[m][n][o][p][q][r]['bias'] = X[m][n][o][p][q][r].pop('b')
                                                                        break 
                                                                except:
                                                                    #import pdb; pdb.set_trace()
                                                                    pass
                                                        except:
                                                            #import pdb; pdb.set_trace()
                                                            pass
                                                except:
                                                    #import pdb; pdb.set_trace()
                                                    pass

    np.save('new.npy',X)
