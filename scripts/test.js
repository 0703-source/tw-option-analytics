const fs=require('fs');
const html=fs.readFileSync(process.argv[2],'utf8');
const wsrc=html.match(/<script type="text\/plain" id="worker-src">([\s\S]*?)<\/script>/)[1];
const main=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m=>m[1]).pop();
new Function(wsrc);new Function(main);console.log('syntax ok');
const RAW=JSON.parse(main.match(/const RAW = (\{[\s\S]*?\});\r?\n/)[1]);
const part=main.slice(main.indexOf('/* ---------- ETL'),main.indexOf('/* ---------- charts'));
const ctx=new Function('RAW',part+'\ningest();buildOptions();return {DB,OPT,bs,brent,bisect};')(RAW);
const {DB,OPT}=ctx;
console.log(DB.log);
if(DB.log.some(s=>s.exc>0)){console.error('ETL exceptions found');process.exit(1)}
console.log('S',OPT.S,OPT.Sdate,'hv20',OPT.hv20.toFixed(3));
for(const e of OPT.exps){const c=OPT.chains[e];const ivs=c.rows.filter(r=>r.C.iv||r.P.iv).length;console.log(e,c.expiry,c.days,'rows',c.rows.length,'atm',c.atm,'atmIV',c.atmIV&&c.atmIV.toFixed(3),'withIV',ivs);
  const near=c.rows.filter(r=>Math.abs(r.K-c.atm)<=600&&r.K%200===0);for(const r of near)console.log('  ',r.K,r.C.settle,r.C.iv&&r.C.iv.toFixed(3),r.C.delta&&r.C.delta.toFixed(3),'|',r.P.settle,r.P.iv&&r.P.iv.toFixed(3));}
// worker
const self={postMessage:m=>{if(m.type==='done'){const a=m.arima,g=m.garch;console.log('ms',m.ms.toFixed(0),'order',a.order,'adf',a.adf,'sigma',a.sigma.toFixed(2),'next',a.next.toFixed(2));console.log('garch',g.params,'nextAnn',g.nextAnn.toFixed(3),'next',g.next.toFixed(2));
  const P=DB.stocks[code].rows.map(r=>r.close),split=Math.floor(P.length*0.8);const act=P.slice(split),prev=P.slice(split-1,-1);
  const rm=p=>Math.sqrt(p.reduce((s,v,i)=>s+(v-act[i])**2,0)/act.length),mape=p=>p.reduce((s,v,i)=>s+Math.abs(v/act[i]-1),0)/act.length;
  const rw=rm(prev);const ens=act.map((_,i)=>(a.preds[i]+g.preds[i])/2);
  for(const [n,p] of [['rw',prev],['arima',a.preds],['garch',g.preds],['ens2',ens]])console.log(n,rm(p).toFixed(2),(mape(p)*100).toFixed(2)+'%','U',(rm(p)/rw).toFixed(3));}
  else if(m.type==='error')console.log('ERR',m.message)}};
new Function('self',wsrc+'\nreturn self;')(self);
const code=process.argv[3]||'2330';
const P=DB.stocks[code].rows.map(r=>r.close);
self.onmessage({data:{P,split:Math.floor(P.length*0.8),id:1}});
