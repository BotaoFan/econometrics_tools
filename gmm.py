#Generalized method of moments
class GMM():
    '''
    mom struction is N*K  array where N is the number of observations and K is the number of moment conditions.
    Preference:
    Je¤rey M. Wooldridge  Econometric Analysis of Cross Section and Panel Data  Chapter 8.3
    Wikipeida 'Generalized method of moments'
    '''
    def __init__(self,endog,exog,instruments,exog_names):
        if len(endog.shape)==1:
            endog=endog.reshape(len(endog),1)
        if endog.shape[0]!=exog.shape[0]:
            raise ValuError('endog is not the same length as exog')
        if exog.shape[0]!=instruments.shape[0]:
            raise ValuError('exog is not the same length as instruments')
        self.endog=endog
        self.exog=exog
        self.instruments=instruments
        self.exog_names=exog_names
        self.nobs=instruments.shape[0]
        self.nmoms=instruments.shape[1]
        self.paras_num=exog.shape[1]
        self.df=self.nobs-self.nmoms
        self.paras=None
        self.weight=None
        self.var_matrix=None
        self.stdvar=None
        self.paras_ttest_pvalue=None
        self.j=None
        self.j_pvalue=None


    def fitgmm(self):
        weight=self.get_init_weight()
        paras=self.get_paras_fomula(weight)
        weight=self.get_weight(paras)
        paras=self.get_paras_fomula(weight)
        self.paras=paras
        self.weight=weight
        return paras,weight

    def get_paras_fomula(self,weight):
        x,y,z,w=self.exog,self.endog,self.instruments,weight
        part1=np.dot(np.dot(np.dot(np.dot(x.T,z),w),z.T),x)
        part2=np.dot(np.dot(np.dot(np.dot(x.T,z),w),z.T),y)
        paras=np.dot(np.linalg.pinv(part1),part2)
        return paras

    def get_init_weight(self):
        z=self.instruments
        return np.linalg.pinv(np.dot(z.T,z)/self.nobs)

    def get_weight(self,paras):
        mom=self.get_mom(paras)
        omiga=np.dot(mom.T,mom)
        weight=np.linalg.pinv(omiga/self.nobs)
        return weight

    def get_mom(self,paras):
        if len(paras.shape)==1:
            paras.reshape(len(paras),1)
        delta=self.get_err(paras)
        mom=self.instruments*delta
        return mom

    def get_err(self,paras):
        return np.dot(self.exog,paras)-self.endog

    def get_J(self):
        if self.paras is None or self.weight is None:
            raise ValuError('Please implement gmm_fit')
        delta=self.get_mom(self.paras)
        delta_mean=delta.mean(axis=0)
        delta_mean=delta_mean.reshape(len(delta_mean),1)
        self.j=self.nobs*np.dot(np.dot(delta_mean.T,self.weight),delta_mean)
        self.j_pvalue=1-(chi2.cdf(self.j,df=self.df))
        return self.j,self.j_pvalue

    def get_stdvar(self):
        if self.paras is None or self.weight is None:
            raise ValuError('Please implement gmm_fit')
        x,z,w=self.exog,self.instruments,(self.weight/self.nobs)
        var_part1=np.dot(np.dot(x.T,z),w)
        var_part2=np.dot(z.T,x)
        self.var_matrix=np.linalg.pinv(np.dot(var_part1,var_part2))
        self.stdvar=np.sqrt([self.var_matrix[i,i] for i in range(self.paras_num)])
        return self.stdvar

    def get_paras_ttest_pvalue(self):
        if self.paras is None or self.weight is None:
            raise ValuError('Please implement gmm_fit')
        if self.stdvar is None:
            stdvar=self.get_stdvar()
        else:
            stdvar=self.stdvar
        paras=self.paras
        paras_ttest_pvalue=[]

        for i in range(self.paras_num):
            if paras[i]>0:
                p_value=2*(1-stats.t.cdf(paras[i]/stdvar[i],df=self.df))
            else:
                p_value=2*stats.t.cdf(paras[i]/stdvar[i],df=self.df)
            paras_ttest_pvalue.append(p_value)
        self.paras_ttest_pvalue=paras_ttest_pvalue
        return self.paras_ttest_pvalue

    def summary(self,save_path_name):
        gmm_result=pd.DataFrame([],index=index_name,columns=columns_name)
        if self.paras is None or self.weight is None:
            raise ValuError('Please implement gmm_fit before get J')

        self.get_J()
        self.get_stdvar()
        self.get_paras_ttest_pvalue()

        para_names=self.exog_names
        paras=self.paras
        stdvar=self.stdvar
        pvalue=self.paras_ttest_pvalue
        j=self.j
        j_pvalue=self.j_pvalue

        if j_pvalue < 0.01:
            star_j = '***'
        elif j_pvalue < 0.05:
            star_j = '**'
        elif j_pvalue < 0.1:
            star_j = '*'
        else:
            star_j = ''

        for i in range(self.paras_num):
            if pvalue[i]<0.01:
                star_para='***'
            elif pvalue[i]<0.05:
                star_para='**'
            elif pvalue[i]<0.1:
                star_para='*'
            else:
                star_para=''
            print "%s : %.6f ( %.6f)%s pvalue=%.6f "%(para_names[i],paras[i],stdvar[i],star_para,pvalue[i])

        print 'Sargan–Hansen J-test(H0: Sum of moments is zero):'
        print 'J is %.8f %s' %(j,star_j)
        print 'Pvalue of J is %0.6f' %j_pvalue
