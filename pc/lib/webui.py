"""The game page: a small web server (Python standard library only) around game.Game.

    py game.py      (starts this, opens the browser)

GET  /           the page
GET  /state      the game as JSON (the page asks twice a second)
GET  /photo.jpg  the last photo with what vision found
POST /action     {"name": "new" | "done" | "dice" | ..., "value": ...}

The terminal works too (for when the page is not open): type
  n = new game   d = done   a = accept   r = read again   c = closest legal move
  52 = dice 5-2  m 24/18 13/11 = pick a move   q = quit
"""
import json
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from game import DEFAULTS, Game

PORT = 8765

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI-bg</title>
<style>
:root{--bg:#f4efe6;--card:#fffaf2;--ink:#2b2622;--mute:#7a6f64;--line:#d9cbb5;
 --robot:#2f5d9e;--human:#b0302a;--accent:#2f6f5e;--warn:#b5651d}
@media (prefers-color-scheme:dark){:root{--bg:#1d1b19;--card:#27241f;--ink:#eee6da;--mute:#a3988a;
 --line:#4a4236;--robot:#5b8fd6;--human:#c8433b;--accent:#5bb39a;--warn:#e09a4f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
 font:15px/1.4 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1180px;margin:0 auto;padding:16px;display:grid;gap:16px}
.pair{display:grid;gap:16px;grid-template-columns:minmax(0,1fr) minmax(0,1fr);align-items:start}
@media (max-width:820px){.pair{grid-template-columns:1fr}}
.pic{position:relative;width:100%;aspect-ratio:450/460;border-radius:8px;overflow:hidden;background:var(--bg);margin-top:8px}
.pic svg,.pic img{position:absolute;inset:0;width:100%;height:100%;display:block;object-fit:fill;margin:0}
#photo,#top{transform:scaleY(-1)}  /* photos upside down too, like the drawing (FLIP, Nadav 14.11) */
button.big{font-size:22px;padding:14px 44px;border-radius:12px}
#edask{position:fixed;z-index:9;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px;box-shadow:0 4px 14px #0004;display:flex;flex-direction:column;gap:6px;font-size:13px}
button.small{font-size:12px;padding:3px 10px;opacity:.75}
.swap{display:inline-flex;align-items:center;gap:8px;margin:4px 0 2px}
.chip{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);border-radius:20px;padding:4px 12px;font-weight:600;background:var(--card)}
.sw{display:inline-block;width:16px;height:16px;border-radius:50%;border:1.5px solid #0008}
.swapbtn{width:36px;height:36px;border-radius:50%;padding:0;font-size:18px;line-height:1;background:transparent;color:var(--ink);border:1px solid var(--line)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}
h1{font-size:18px;margin:0 0 4px}.mute{color:var(--mute);font-size:13px}
#prompt{font-size:20px;font-weight:600;margin:6px 0 10px}
button,select,input{font:inherit;border-radius:8px;border:1px solid var(--line);padding:8px 12px;
 background:var(--card);color:var(--ink)}
button{cursor:pointer;background:var(--accent);color:#fff;border:0;font-weight:600}
button.sec{background:transparent;color:var(--ink);border:1px solid var(--line)}
#asks{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.dice{display:inline-flex;gap:8px;margin-left:10px;vertical-align:middle}.die{display:inline-block;width:38px;height:38px}
.die.rolling{animation:tumble .35s linear infinite}
@keyframes tumble{0%{transform:rotate(0) translateY(0)}25%{transform:rotate(90deg) translateY(-6px)}
 50%{transform:rotate(180deg) translateY(0)}75%{transform:rotate(270deg) translateY(-4px)}100%{transform:rotate(360deg)}}
svg{width:100%;height:auto;display:block}
.warn{color:var(--warn);font-size:13px}
table{width:100%;border-collapse:collapse;font-size:13px}td{padding:3px 4px;border-bottom:1px solid var(--line)}
#photo{display:none}.photo2{width:100%;border-radius:8px;display:none;margin-top:8px}
.grid{display:grid;grid-template-columns:auto 1fr;gap:6px 10px;align-items:center}
.stats{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:13px;margin-top:6px}
</style></head><body><main>
<section class="card">
  <h1>AI-bg <span class="mute" id="sim"></span></h1>
  <div id="prompt">…</div>
  <div id="asks"></div>
  <div class="stats"><b id="turn"></b><span id="offs"></span></div>
  <div class="swap" title="Which colour is whose">
   <span class="chip"><i class="sw" style="background:var(--robot)"></i><span id="swr">Robot</span></span>
   <button class="swapbtn" id="swapbtn" title="Swap the colours" aria-label="Swap the colours">⇄</button>
   <span class="chip"><i class="sw" style="background:var(--human)"></i><span id="swh">You</span></span>
  </div>
  <div id="coach" class="mute"></div>
  <div id="warns"></div>
</section>
<section class="pair">
 <div class="card"><h1>The game</h1>
  <div class="pic"><svg id="board" viewBox="-20 -20 450 460" preserveAspectRatio="none"></svg></div>
  <div class="mute" style="margin-top:6px">Same way round as the camera's view next to it (both shown upside down). Numbers are the columns.</div></div>
 <div class="card"><h1>What the camera sees</h1>
  <div class="pic"><img id="photo" alt="top view" style="display:none"></div>
  <div class="mute" style="margin-top:6px">The board from above, circles = checkers found.</div></div>
</section>
<section style="display:grid;gap:16px;align-content:start">
  <details class="card"><summary class="mute">Original camera photo</summary>
   <img id="top" class="photo2" alt="raw photo"></details>
  <details class="card"><summary class="mute">Settings (used from the next New game)</summary>
   <div class="grid" id="settings" style="margin-top:8px"></div></details>
  <table id="hist" hidden></table>
</section></main>
<script>
const OPTS={mode:["robot_vs_human","robot_vs_robot","watch"],engine:["gnubg","simple","random"],
 dice:["digital","typed","camera"],human_move:["done_button","auto","robot_moves"],
 vision:["learned","norm"],coach:[false,true],first:["roll","robot","human"],board:["auto","real","sim"]};
const NAMES={mode:"Mode",engine:"Engine",dice:"Dice",human_move:"Your move",vision:"Vision",
 coach:"Coach",first:"Who starts",board:"Board"};
const BTN={new:"New game",done:"Done",accept:"Accept",reread:"Read again",closest:"Take closest legal move",
 learn:"Learn colours",arrange:"Robot arranges the board",rolled:"Rolled"};
let S=null, built=false, lastPhoto=0, ED=null, EDC="robot";
// -- the board is drawn as the corrected camera photo shows it: frame mm (x, y from the near side)
// -> magnet mm (x right, y down) through the layout's own map (s.frame), 1 unit = 1 mm --
// FLIP: the drawing is shown upside down compared with the camera's view (Nadav 14.11)
const FLIP=true;
function P(x,y){const A=S.frame,b=A[1][0]*x+A[1][1]*y+A[1][2];return [A[0][0]*x+A[0][1]*y+A[0][2],FLIP?420-b:b]}
function Q(x,y){const [a,b]=P(x,y);return `${a.toFixed(1)},${b.toFixed(1)}`}
function toFrame(mx,my){if(FLIP)my=420-my;const A=S.frame,d=A[0][0]*A[1][1]-A[0][1]*A[1][0],u=mx-A[0][2],v=my-A[1][2];
 return [(A[1][1]*u-A[0][1]*v)/d,(-A[1][0]*u+A[0][0]*v)/d]}
function txt(x,y,t,size,fill,bold){const [a,b]=P(x,y);
 return `<text x="${a}" y="${b+size*0.35}" font-size="${size}" ${bold?'font-weight="bold"':''} text-anchor="middle" fill="${fill}">${t}</text>`}
function disc(x,y,side){const [a,b]=P(x,y);return `<circle cx="${a}" cy="${b}" r="12.5" fill="var(--${side})" stroke="#0008" stroke-width="1.2"/>`}
function base(s,edit){let h=`<rect x="-20" y="-20" width="450" height="460" rx="10" fill="var(--card)" stroke="${edit?'#e0a000':'var(--line)'}" stroke-width="${edit?4:1}"/>`;
 if(edit)h+=`<rect x="-20" y="-20" width="450" height="460" rx="10" fill="#e0a000" opacity=".10"/>`;
 const by=s.bar_y;h+=`<polygon points="${Q(0,by-13)} ${Q(410,by-13)} ${Q(410,by+13)} ${Q(0,by+13)}" fill="var(--line)" opacity=".35"/>`;
 for(const [x,y,n] of s.labels){const left=x<205;
  const pts=left?`${Q(x-13,y+13)} ${Q(x-13,y-13)} ${Q(x+150,y)}`:`${Q(x+13,y+13)} ${Q(x+13,y-13)} ${Q(x-150,y)}`;
  h+=`<polygon points="${pts}" fill="${n%2?'#c9a77c':'#8f6a43'}" opacity=".45"/>`;
  h+=txt(left?x+162:x-162,y,n,11,"var(--mute)")}
 return h}
function edAdd(p,side){if(p===0||p===25){if(side==="robot")ED[0]++;else ED[25]--}
 else if(side==="robot"){if(ED[p]<0)ED[p]=1;else ED[p]++}else{if(ED[p]>0)ED[p]=-1;else ED[p]--}drawEdit(S)}
function edAsk(ev,p){// a new column (or the bar): which colour to add?
 let m=document.getElementById("edask");if(m)m.remove();
 m=el("div",{id:"edask",class:"edask"});m.style.left=Math.min(ev.clientX,innerWidth-190)+"px";m.style.top=ev.clientY+"px";
 m.append(el("div",{text:"Which colour to add?"}));
 for(const side of ["robot","human"]){const b=el("button",{text:side==="robot"?"Robot":"You",onclick:e=>{e.stopPropagation();m.remove();edAdd(p,side)}});
  b.style.borderLeft="14px solid var(--"+side+")";m.append(b)}
 document.body.append(m)}
document.addEventListener("click",e=>{const m=document.getElementById("edask");if(m&&!m.contains(e.target)&&e.target.id!=="board"&&!e.target.closest("#board"))m.remove()});
function edPlace(ev){if(!ED)return;const old=document.getElementById("edask");if(old)old.remove();
 const svg=document.getElementById("board");const pt=svg.createSVGPoint();
 pt.x=ev.clientX;pt.y=ev.clientY;const q=pt.matrixTransform(svg.getScreenCTM().inverse());
 const [fx,y]=toFrame(q.x,q.y),left=fx<205;
 // the bar: a click on a checker takes it off, anywhere else on the bar asks which colour to add
 if(Math.abs(fx-205)<22&&Math.abs(y-S.bar_y)<BARP*5){const st=Math.round((y-S.bar_y)/BARP),k=st>0?2*st:st<0?-2*st-1:0;
  const rob=k%2===0,i=rob?k/2:(k-1)/2,have=rob?ED[0]:-ED[25];
  if(i<have&&Math.abs(y-(S.bar_y+barStep(k)*BARP))<14){if(rob)ED[0]--;else ED[25]++;drawEdit(S)}
  else edAsk(ev,rob?0:25);return}
 let best=null;for(const [x,ly,n] of S.labels){if((x<205)!==left)continue;const d=Math.abs(ly-y);if(!best||d<best[0])best=[d,25-n,x]}
 if(!best)return;const p=best[1],v=ED[p],n=Math.abs(v);
 if(!n){edAsk(ev,p);return}                                  // an empty column: ask the colour
 const step=n>1?Math.min(26,138/(n-1)):0,dist=(fx-best[2])*(left?1:-1);
 if(dist<(n-1)*step+13){ED[p]+=v>0?-1:1}                      // on a checker: take it off (whichever colour)
 else ED[p]+=v>0?1:-1;                                         // on the empty part of the column: one more of its colour
 drawEdit(S)}
function offOf(b){let r=0,h=0;for(const v of b){if(v>0)r+=v;else h-=v}return [15-r,15-h]}
function stack(x,y,dir,v,side,vert){// checkers side by side from the edge toward the middle, like a real board
 const n=Math.abs(v);if(!n)return "";const step=n>1?Math.min(vert?29:26,(vert?100:138)/(n-1)):0;let h="";
 for(let i=0;i<n;i++)h+=vert?disc(x,y+dir*i*step,side):disc(x+dir*i*step,y,side);
 if(n>6)h+=vert?txt(x,y+dir*(n-1)*step,n,13,ink(side),true):txt(x+dir*(n-1)*step,y,n,13,ink(side),true);
 return h}
const BARP=29;// hit checkers of both colours: one column on the centre line, out from the middle (see layout.py)
function barStep(k){return Math.floor((k+1)/2)*(k%2?-1:1)}
function drawEdit(s){const svg=document.getElementById("board");
 let h=base(s,true)+txt(205,s.labels[11][1],"EDITING",13,"#c08000",true);
 for(const [x,y,n] of s.labels){const v=ED[25-n];if(v)h+=stack(x,y,x<205?1:-1,v,v>0?"robot":"human")}
 for(let i=0;i<ED[0];i++)h+=disc(205,s.bar_y+barStep(2*i)*BARP,"robot");
 for(let i=0;i<-ED[25];i++)h+=disc(205,s.bar_y+barStep(2*i+1)*BARP,"human");
 const o=offOf(ED);h+=txt(205,35,`off: robot ${o[0]} · you ${o[1]}`,12,"var(--mute)");
 svg.innerHTML=h}
// -- dice: pips, a short tumble and a rattle when new dice appear --
const PIPS={1:[[2,2]],2:[[1,1],[3,3]],3:[[1,1],[2,2],[3,3]],4:[[1,1],[3,1],[1,3],[3,3]],
 5:[[1,1],[3,1],[2,2],[1,3],[3,3]],6:[[1,1],[3,1],[1,2],[3,2],[1,3],[3,3]]};
function face(v){return `<svg viewBox="0 0 40 40" width="38" height="38"><rect x="1" y="1" width="38" height="38" rx="8" fill="#fff" stroke="#999"/>`+
 (PIPS[v]||[]).map(([i,j])=>`<circle cx="${i*10}" cy="${j*10}" r="3.6" fill="#222"/>`).join("")+`</svg>`}
let lastDice, rollEnd=0, AC=null;
function rolling(){return Date.now()<rollEnd}
function rattle(){try{AC=AC||new (window.AudioContext||window.webkitAudioContext)();const t0=AC.currentTime;
 for(let k=0;k<7;k++){const t=t0+k*0.07+Math.random()*0.03,n=AC.createBufferSource(),b=AC.createBuffer(1,AC.sampleRate*0.03,AC.sampleRate),
  d=b.getChannelData(0);for(let i=0;i<d.length;i++)d[i]=(Math.random()*2-1)*Math.pow(1-i/d.length,3);
  const f=AC.createBiquadFilter();f.type="bandpass";f.frequency.value=1800+Math.random()*1800;f.Q.value=2;
  const g=AC.createGain();g.gain.value=0.5*(1-k/9);n.buffer=b;n.connect(f).connect(g).connect(AC.destination);n.start(t)}}catch(e){}}
function roll(){rollEnd=Date.now()+700;rattle();const t=setInterval(()=>{const ds=document.querySelectorAll(".die");
 for(const e of ds){if(rolling())e.innerHTML=face(1+Math.floor(Math.random()*6));else{e.innerHTML=face(+e.dataset.v);e.classList.remove("rolling")}}
 if(!rolling())clearInterval(t)},80)}
document.addEventListener("pointerdown",()=>{try{AC=AC||new (window.AudioContext||window.webkitAudioContext)();AC.resume()}catch(e){}},{once:true});
function post(a){return fetch("/action",{method:"POST",body:JSON.stringify(a)})}
function el(t,attrs={},kids=[]){const e=document.createElement(t);for(const k in attrs){
 if(k=="text")e.textContent=attrs[k];else if(k.startsWith("on"))e[k]=attrs[k];else e.setAttribute(k,attrs[k])}
 for(const c of kids)e.append(c);return e}
function buildSettings(s){const g=document.getElementById("settings");g.innerHTML="";
 for(const k in OPTS){const sel=el("select",{onchange:()=>{let v=sel.value;if(v==="true")v=true;if(v==="false")v=false;
   post({name:"settings",values:{[k]:v}})}});
  for(const o of OPTS[k]){const op=el("option",{value:String(o),text:String(o).replaceAll("_"," ")});
   if(String(s[k])===String(o))op.selected=true;sel.append(op)}
  g.append(el("label",{text:NAMES[k]}),sel)}}
function drawBoard(s){const svg=document.getElementById("board");
 let h=base(s,false)+txt(205,35,"bear-off",11,"var(--mute)");
 for(const [x,y,side] of s.pieces)h+=disc(x,y,side);
 svg.innerHTML=h}
function ink(side){const c=getComputedStyle(document.documentElement).getPropertyValue("--"+side).trim();
 const m=/^#([0-9a-f]{6})$/i.exec(c);if(!m)return side==="robot"?"#000":"#fff";const n=parseInt(m[1],16);
 return (0.299*(n>>16)+0.587*((n>>8)&255)+0.114*(n&255))>150?"#000":"#fff"}
const DEF={robot:"#2f5d9e",human:"#b0302a"};let SWAP=false;
try{SWAP=localStorage.getItem("swapSides")==="1"}catch(e){}
function edText(v){const c=getComputedStyle(document.documentElement).getPropertyValue("--"+v).trim();
 return `click adds ${NAMEOF[c]||"a"} checker (${v==="robot"?"robot":"you"})`}
const NAMEOF={"#f2f2ea":"white","#1c1c1c":"black","#2f5d9e":"blue","#b0302a":"red"};
function paintSides(c){const r=document.documentElement.style;
 if(!c&&!SWAP){r.removeProperty("--robot");r.removeProperty("--human")}
 else{const b=c||DEF;r.setProperty("--robot",b[SWAP?"human":"robot"]);r.setProperty("--human",b[SWAP?"robot":"human"])}
 const g=k=>NAMEOF[getComputedStyle(document.documentElement).getPropertyValue("--"+k).trim()]||"";
 document.getElementById("swr").textContent="Robot"+(g("robot")?": "+g("robot"):"");
 for(const v of ["robot","human"]){const o=document.querySelector(".ed-"+v);if(o)o.textContent=edText(v)}
 document.getElementById("swh").textContent="You"+(g("human")?": "+g("human"):"")}
document.getElementById("swapbtn").onclick=()=>{SWAP=!SWAP;try{localStorage.setItem("swapSides",SWAP?"1":"0")}catch(e){}
 paintSides(S&&S.colours);if(S){if(ED)drawEdit(S);else drawBoard(S)}};
function render(s){S=s;paintSides(s.colours);if(!built){buildSettings(s.settings);built=true}
 document.getElementById("sim").textContent=s.sim?"(simulator)":"";
 const p=document.getElementById("prompt");p.textContent=s.prompt;
 const dk=JSON.stringify(s.dice||null);if(dk!==lastDice){if(lastDice!==undefined&&s.dice)roll();lastDice=dk}
 if(s.dice){const d=el("span",{class:"dice"});for(const v of s.dice){const e=el("span",{class:"die"+(rolling()?" rolling":"")});
  e.dataset.v=v;e.innerHTML=face(rolling()?1+Math.floor(Math.random()*6):v);d.append(e)}p.append(d)}
 const a=document.getElementById("asks");const keep=document.activeElement&&a.contains(document.activeElement);
 if(!keep||a.dataset.k!==JSON.stringify(s.asks)+s.legal.length){a.innerHTML="";a.dataset.k=JSON.stringify(s.asks)+s.legal.length;
  for(const k of s.asks){
   if(k==="dice_input"){const i=el("input",{placeholder:"dice, e.g. 5 2",size:8});
    a.append(i,el("button",{text:"OK",onclick:()=>post({name:"dice",value:i.value})}))}
   else if(k==="edit"){
    a.append(el("div",{class:"mute",text:"Fix the reading: click a checker to take it off; click empty space in a column to add one of its colour; click an empty column or the bar to choose the colour."}),
     el("button",{text:"Use this board",onclick:()=>post({name:"board",value:ED})}),
     el("button",{class:"sec",text:"Back to the camera's reading",onclick:()=>{ED=S.seen.slice();drawEdit(S)}}))}
   else if(k==="move_pick"){const sel=el("select");for(const m of s.legal)sel.append(el("option",{text:m}));
    a.append(sel,el("button",{text:"Play this",onclick:()=>post({name:"move",value:sel.value})}))}
   else if(k==="new")continue;
   else a.append(el("button",{class:k==="done"?"big":"",text:BTN[k]||k,onclick:()=>post({name:k})}))}
  a.append(el("button",{class:"sec small",text:"New game",onclick:()=>post({name:"new"})}))}
 paintSides(s.colours);
 document.getElementById("turn").textContent=s.winner?"":(s.turn==="robot"?"🤖 Robot's turn":s.turn?"🙂 Your turn":"");
 document.getElementById("offs").textContent=(s.off[0]||s.off[1])?`Taken off: robot ${s.off[0]}/15, you ${s.off[1]}/15`:"";
 document.getElementById("coach").textContent=s.coach||"";
 const w=document.getElementById("warns");w.innerHTML="";for(const t of s.warnings)w.append(el("div",{class:"warn",text:"⚠ "+t}));
 if(s.asks.includes("edit")&&s.seen){if(!ED)ED=s.seen.slice();drawEdit(s)}else{ED=null;drawBoard(s)}
 const t=document.getElementById("hist");t.innerHTML="";
 for(const m of s.history.slice().reverse())t.append(el("tr",{},[el("td",{text:m.n}),el("td",{text:m.who}),
  el("td",{text:m.dice.join("-")}),el("td",{text:m.move}),el("td",{class:"mute",text:m.note||""})]));
 if(s.has_photo&&s.photo_t!==lastPhoto){lastPhoto=s.photo_t;const im=document.getElementById("photo");
  im.src="/top.jpg?t="+s.photo_t;im.style.display="block";
  const tp=document.getElementById("top");tp.src="/photo.jpg?t="+s.photo_t;tp.style.display="block"}}
async function tick(){try{render(await (await fetch("/state")).json())}catch(e){}setTimeout(tick,500)}
document.getElementById("board").addEventListener("click",edPlace);
document.addEventListener("keydown",ev=>{// Enter = Done, when Done is offered and nothing is being typed
 if(ev.key!=="Enter"||!S||!S.asks.includes("done"))return;
 const t=document.activeElement&&document.activeElement.tagName;if(t==="INPUT"||t==="SELECT"||t==="BUTTON"||t==="TEXTAREA")return;
 ev.preventDefault();post({name:"done"})});
tick();
</script></body></html>"""


def make_handler(game):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype, code=200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/state":
                self._send(json.dumps(game.snapshot()).encode(), "application/json")
            elif path == "/photo.jpg" and game.photo:
                self._send(game.photo, "image/jpeg")
            elif path == "/top.jpg" and getattr(game, "photo_top", None):
                self._send(game.photo_top, "image/jpeg")
            else:
                self._send(b"not found", "text/plain", 404)

        def do_POST(self):
            if self.path != "/action":
                return self._send(b"not found", "text/plain", 404)
            n = int(self.headers.get("Content-Length", 0))
            try:
                game.post(json.loads(self.rfile.read(n) or b"{}"))
            except ValueError:
                return self._send(b"bad json", "text/plain", 400)
            self._send(b"{}", "application/json")
    return H


def terminal(game):
    import dice
    keys = {"n": "new", "d": "done", "a": "accept", "r": "reread", "c": "closest", "l": "learn",
            "g": "arrange", "o": "rolled"}
    while True:
        try:
            t = input().strip()
        except EOFError:
            return
        if t == "q":
            game.post({"name": "stop"})
            return
        if t in keys:
            game.post({"name": keys[t]})
        elif t.startswith("m "):
            game.post({"name": "move", "value": t[2:].strip()})
        elif dice.parse(t):
            game.post({"name": "dice", "value": t})
        elif t:
            print("n=new d=done a=accept r=read again c=closest l=learn g=arrange o=rolled 52=dice m <move> q=quit")


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "localhost"


def main(settings=None, open_browser=True, port=PORT):
    game = Game(settings)
    srv = ThreadingHTTPServer(("0.0.0.0", port), make_handler(game))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://localhost:{port}"
    print(f"Game page: {url}   (phone on the same WiFi: http://{lan_ip()}:{port})")
    print("Terminal: n=new game d=done 52=dice q=quit")
    if open_browser:
        webbrowser.open(url)
    threading.Thread(target=terminal, args=(game,), daemon=True).start()
    try:
        game.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
        if game.b is not None:
            try:
                game.b.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()


__all__ = ["main", "DEFAULTS"]
