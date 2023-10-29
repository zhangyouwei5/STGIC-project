import os
from STGIC.utils import set_seed
from STGIC.preprocess import *
from STGIC.AGC import *
from STGIC.DCF import *
import seaborn as sns


os.chdir('/home/zhangchen/validate')
n_components_agc=50
n_components=15
use_high_agc=False
use_high=True
agc_dims=26

nChannel=100
output_dim=100
pretrain_lr=0.05
lr=0.01
step_kl=0.78
step_con=0.62
step_con1=0.58
step_ce=0.71
q_cut=0.5
mod3_ratio=0.7

seed=0
device=torch.device("cuda:0")

pid='mouse_postbrain'
h5ad_file = '%s.h5ad' % pid
img,coord_lattice,mask,adj,feat_agc,adata=generate_input_visium(h5ad_file,n_components_agc,n_components,use_high_agc,use_high)
n_clusters=20
label=None
adj_normalized = normalize_adj(adj)
adj_normalized = 1/3*torch.eye(adj_normalized.shape[0]).to(device)+2/3*adj_normalized
optimal_power,y_pred_init,kmeans_feature,pre_nmi,pre_ari=pre_cluster(adj_normalized,feat_agc,n_clusters,label,top_num=agc_dims,max_iter=30)


coord_x=torch.tensor(coord_lattice[:,0]).to(torch.long).to(device)
coord_y=torch.tensor(coord_lattice[:,1]).to(torch.long).to(device)
set_seed(0)
model3 = Dilate_MyNet5_visium(img.shape[2],output_dim,nChannel,n_clusters,3,2).to(device)
model2 = Dilate_MyNet5_visium(img.shape[2],output_dim,nChannel,n_clusters,2,2).to(device)
nmi,ari,y_pred_last,emb,final_epoch_idx=dilate2_train5_visium(model3,model2,coord_x,coord_y,n_clusters,step_kl,step_con,step_con1,step_ce,label,img,mask,y_pred_init,200,400,tol=1e-3,update_interval=4,q_cut=q_cut,cat_tol=1e-3,mod3_ratio=mod3_ratio)


adata.obs['y_pred']=[str(int(i)) for i in y_pred_last]
adata.obs['y_pred_init']=[str(int(i)) for i in y_pred_init]
adata.obs['x_pixel']=list(adata.obsm['spatial'][:,0])
adata.obs['y_pixel']=list(-adata.obsm['spatial'][:,1])
sc.pl.spatial(adata,color='y_pred')


adata.obsm['rep']=emb
sc.pp.neighbors(adata,use_rep='rep',random_state=0)
sc.tl.umap(adata,random_state=0)
sc.pl.umap(adata, color=['y_pred'])






sc.tl.rank_genes_groups(adata, 'y_pred', method='wilcoxon')
for i in ['8','18','2','19','13','5']:
    a=sc.get.rank_genes_groups_df(adata,i)
    b=a[(a.pvals_adj<0.01)&(a.logfoldchanges>2)]
    b.sort_values(by='scores',inplace=True,ascending=False)
    sc.pl.spatial(adata, img_key="hires", color=list(b.names)[:10])

for i in ['3']:
    a=sc.get.rank_genes_groups_df(adata,'3')
    b=a[(a.pvals_adj<0.01)&(a.logfoldchanges>3)]
    b.sort_values(by='logfoldchanges',inplace=True,ascending=False)
    print(list(b.names))
    sc.pl.spatial(adata, img_key="hires", color=list(b.names)[:11])


svg_pool=[]
for i in range(n_clusters):
    tmp=sc.get.rank_genes_groups_df(adata,str(i))
    tmp1=tmp[(tmp.pvals_adj<0.01)&(tmp.logfoldchanges>2)]
    if tmp1.shape[0]!=0:
        svg_pool.append(tmp1)
svg=pd.concat(svg_pool,0)
svg_stgic=list(set(list(svg.names.values)))
stgic_svg_idx=[list(adata.var_names).index(i) for i in svg_stgic]


spark_gene=pd.read_table('most_postbrain_sparkx_svg.txt')
spark_gene=spark_gene.drop(spark_gene[spark_gene.isnull().values==True].index,0)
spark_gene.sort_values(by='adjustedPval',inplace=True,ascending=False)
svg_spark=list(set(list(spark_gene.index)[:1000]))
var_names=list(adata.var_names)
spark_svg_idx=[]
for i in svg_spark:
    for j in range(len(var_names)):
        if i==var_names[j]:
            spark_svg_idx.append(j)
            break
    
print(len(svg_stgic),len(set(svg_stgic)&set(svg_spark)))

sc.pp.neighbors(adata)

stgic_moran = sc.metrics.morans_i(adata.obsp["distances"], adata.X[:,stgic_svg_idx].T)
print('mean and median moran I corresponding to stgic are', np.mean(stgic_moran),np.median(stgic_moran))


spark_moran = sc.metrics.morans_i(adata.obsp["distances"], adata.X[:,spark_svg_idx].T)
print('mean and median moran I corresponding to SPARK-X are', np.mean(spark_moran),np.median(spark_moran))


data = pd.DataFrame({ 'STGIC':pd.Series(stgic_moran),"SPARK-X": pd.Series(spark_moran)})
ax = sns.boxplot(data=data,showfliers=0,color='white',
                 medianprops={'color': 'red', 'linewidth': '1.5'},
                 showmeans=True,
                 meanline=True,
                 meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'}
                 )
ax.set_ylabel("Moran's I")
ax.set_title('Comparison of STGIC with SPARK-X')
ax.get_figure().savefig('mouse_postbrain_svg_boxplot.png', dpi = 600)

