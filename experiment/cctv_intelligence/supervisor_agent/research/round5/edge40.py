import random, json, edge
random.seed(5)
pts=[]
# 40 stays: dense around Press 15 / Bench 15, many near edges
while len(pts)<40:
    x=round(random.uniform(3.8,10.2),2); y=round(random.uniform(1.95,6.2),2)
    a,b,c,d=edge.OBJ['Bench 15']; A,B,C,D=edge.OBJ['Press 15']
    if a<=x<=c and b<=y<=d: continue
    if A<=x<=C and B<=y<=D: continue
    pts.append((x,y))
edge.P[:]=pts
edge.TRUTH[:]=[dict(nearest=edge.nearest(x,y),side=edge.side15(x,y),reach=edge.reach_press15(x,y),part=edge.part15(x,y)) for x,y in pts]
edge.hh=lambda i:f'10:{i//4:02d}:{(i%4)*15:02d}'
import os; os.makedirs('out40',exist_ok=True)
edge.image('out40/plan.png')
Q=edge.Q.replace('s1–s12','s1–s40').replace('"s12":{...}','"s40":{...}')
for k,v in {'rel':edge.fmt_rel(),'raw':edge.fmt_raw(),'img':'DATA: one image (attached): floor plan with the stays drawn as numbered dots. A metre grid is drawn.'}.items():
    open(f'out40/{k}.txt','w').write(edge.BRIEF+'\n\nDATA:\n'+v+'\n\n'+Q)
json.dump(edge.TRUTH,open('truth40.json','w'))
print(sum(1 for t in edge.TRUTH if t['reach']=='yes'),'reach yes;', {s:sum(1 for t in edge.TRUTH if t['side']==s) for s in set(t['side'] for t in edge.TRUTH)})
