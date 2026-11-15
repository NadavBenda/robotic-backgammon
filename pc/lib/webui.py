"""The game page: a small web server (Python standard library only) around game.Game.

    py game.py      (starts this, opens the browser)

GET  /           the page
GET  /state      the game as JSON (the page asks twice a second)
GET  /top.jpg    the last photo as a top view, the known board circled, the robot's paths
GET  /photo.jpg  the last photo as the camera took it (upside down, like the drawing)
POST /action     {"name": "new" | "done" | "dice" | ..., "value": ...}

The terminal works too (for when the page is not open): type
  n = new game   s = start   g = robot arranges from the trays   d = done   r = read again
  a = accept     o = rolled  52 = dice 5-2     m 24/18 13/11 = pick a move   q = quit
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
:root{--bg:#12100e;--card:#1d1915;--card2:#262019;--ink:#f3ebdd;--mute:#a59882;--line:#3a3128;
 --gold:#e2b053;--accent:#e2b053;--warn:#e09a4f;--felt:#1d4d3f;--wood:#4a2f1c}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(1200px 700px at 50% -10%,#2a2119 0,var(--bg) 60%) fixed,var(--bg);color:var(--ink);
 font:15px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif;min-height:100vh}
main{max-width:1240px;margin:0 auto;padding:14px 16px 28px;display:grid;gap:14px}
.card{background:linear-gradient(180deg,var(--card2),var(--card));border:1px solid var(--line);border-radius:16px;padding:14px;
 box-shadow:0 10px 30px #0006}
h2{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--mute);margin:0 0 8px;font-weight:600}
.mute{color:var(--mute);font-size:13px}
/* -- top bar: name, the two players -- */
.top{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.logo{font-size:22px;font-weight:800;letter-spacing:.04em}.logo b{color:var(--gold)}
#sim{font-size:11px;color:var(--mute);border:1px solid var(--line);border-radius:10px;padding:1px 8px;margin-left:8px;vertical-align:middle}
#sim:empty{display:none}
.players{margin-left:auto;display:flex;align-items:center;gap:10px}
.player{display:flex;align-items:center;gap:10px;padding:6px 14px 6px 6px;border-radius:40px;border:1.5px solid var(--line);
 background:var(--card);transition:border-color .3s,box-shadow .3s}
.player.on{border-color:var(--gold);box-shadow:0 0 0 3px #e2b05333,0 0 22px #e2b05344}
.player .face{width:46px;height:46px;border-radius:50%;background:radial-gradient(circle at 35% 30%,#3f7a66,#17382e);display:grid;place-items:center;flex:none}
.player .face svg{width:32px;height:32px}
.player .nm{font-weight:700;line-height:1.1}.player .st{font-size:12px;color:var(--mute);white-space:nowrap}
.swapbtn{width:34px;height:34px;border-radius:50%;padding:0;font-size:17px;line-height:1;background:transparent;color:var(--mute);border:1px solid var(--line)}
.swapbtn:hover{color:var(--gold);border-color:var(--gold)}
/* -- the stage: what to do now, the buttons, the dice pinned at the right -- */
.stage{display:grid;grid-template-columns:minmax(0,1fr) 96px;gap:16px;align-items:center;min-height:196px;position:relative;overflow:hidden}
.stage::before{content:"";position:absolute;left:0;top:0;height:3px;width:40%;background:linear-gradient(90deg,transparent,var(--gold),transparent);
 opacity:0;transform:translateX(-100%)}
.stage.busy::before{opacity:1;animation:slide 1.1s linear infinite}
@keyframes slide{to{transform:translateX(250%)}}
#prompt{font-size:30px;font-weight:800;line-height:1.15;margin:0 0 14px;display:flex;align-items:center;gap:12px}
.spin{width:22px;height:22px;border-radius:50%;border:3px solid #ffffff22;border-top-color:var(--gold);animation:rot .8s linear infinite;flex:none;display:none}
.stage.busy .spin{display:block}
@keyframes rot{to{transform:rotate(360deg)}}
#asks{display:flex;flex-wrap:wrap;gap:10px;align-items:center}
.stage.busy #asks{opacity:.45;pointer-events:none;filter:grayscale(.6)}
#coach{margin-top:8px}
button,select,input{font:inherit;border-radius:10px;border:1px solid var(--line);padding:9px 14px;background:var(--card);color:var(--ink)}
button{cursor:pointer;background:linear-gradient(180deg,#f0c56d,#d79f3c);color:#231a0c;border:0;font-weight:700;
 box-shadow:0 2px 0 #8a6220,0 6px 14px #0005;transition:transform .08s,filter .15s}
button:hover{filter:brightness(1.08)}button:active{transform:translateY(2px);box-shadow:0 0 0 #8a6220}
button.sec{background:transparent;color:var(--ink);border:1px solid var(--line);box-shadow:none;font-weight:600}
button.sec:hover{border-color:var(--gold);color:var(--gold)}
button.big{font-size:24px;padding:14px 54px;border-radius:14px}
button.small{font-size:12px;padding:4px 10px;opacity:.7}
.hint{flex-basis:100%;color:var(--mute);font-size:13px}
.dicecol{display:flex;flex-direction:column;gap:12px;align-items:center;justify-content:center}
.die{display:block;width:84px;height:84px;filter:drop-shadow(0 6px 8px #0008)}
.die.none{opacity:.13;filter:none}
.die.rolling{animation:tumble .35s linear infinite}
@keyframes tumble{0%{transform:rotate(0) translateY(0)}25%{transform:rotate(90deg) translateY(-6px)}
 50%{transform:rotate(180deg) translateY(0)}75%{transform:rotate(270deg) translateY(-4px)}100%{transform:rotate(360deg)}}
/* -- the board and the camera -- */
.pair{display:grid;gap:14px;grid-template-columns:minmax(0,1fr) minmax(0,1fr);align-items:start}
@media (max-width:860px){.pair{grid-template-columns:1fr}#prompt{font-size:22px}.players{margin-left:0}
 .stage{grid-template-columns:minmax(0,1fr) 68px;min-height:150px}.die{width:60px;height:60px}button.big{padding:12px 30px;font-size:20px}}
.pic{position:relative;width:100%;aspect-ratio:450/460;border-radius:12px;overflow:hidden;background:#0b0a09}
.pic svg,.pic img{position:absolute;inset:0;width:100%;height:100%;display:block;object-fit:fill;margin:0}
#edge{display:flex;align-items:center;gap:10px;margin:0 0 10px;font-size:12.5px;font-weight:700;white-space:nowrap}
#edge[hidden]{display:none}
.edgebar{flex:1;height:14px;border-radius:8px;overflow:hidden;border:1px solid #6b6256;position:relative}
.edgebar i{position:absolute;left:0;top:0;bottom:0;transition:width .8s ease}
.edgebar::after{content:"";position:absolute;left:50%;top:0;bottom:0;width:2px;margin-left:-1px;background:var(--gold);opacity:.9}
#info{margin-top:10px;font-weight:600;color:var(--gold);min-height:21px}
#log{margin-top:4px;font-size:12.5px;color:var(--mute);display:grid;gap:1px}
#log div:first-child{color:var(--ink)}
#log b{display:inline-block;min-width:62px;font-weight:600}
.photo2{width:100%;border-radius:8px;display:none;margin-top:8px}
.grid{display:grid;grid-template-columns:auto 1fr;gap:6px 10px;align-items:center}
.low{display:grid;gap:10px}
.tog{padding:4px 8px;font-size:15px;line-height:1}
body.clean .low,body.clean #newbtn,body.clean #log,body.clean .camnote,body.clean #sim{display:none}
details.card{padding:10px 14px}summary{cursor:pointer}
.warn{color:var(--warn);font-size:13px}
#edask{position:fixed;z-index:9;background:var(--card2);border:1px solid var(--line);border-radius:10px;padding:8px;box-shadow:0 4px 14px #0008;display:flex;flex-direction:column;gap:6px;font-size:13px}
/* -- the end of the game -- */
.stage.won{border-color:var(--gold);box-shadow:0 0 0 2px #e2b05355,0 10px 40px #e2b05333}
.stage.won #prompt{font-size:44px;color:var(--gold)}
.stage.lost #prompt{font-size:40px;color:#b9b0a2}
#confetti{position:fixed;inset:0;pointer-events:none;overflow:hidden;z-index:20}
#confetti i{position:absolute;top:-20px;width:10px;height:16px;border-radius:2px;animation:fall linear forwards}
@keyframes fall{to{transform:translateY(110vh) rotate(720deg)}}
</style></head><body><main>
<header class="top">
 <div class="logo">AI<b>·</b>bg<span id="sim"></span></div>
 <button class="sec small" id="newbtn" hidden>New game</button>
 <button class="sec small tog" id="sndbtn" title="Sounds: dice, your turn, the end of the game"></button>
 <button class="sec small tog" id="cleanbtn" title="Video mode: only the game, nothing else on the page"></button>
 <div class="players" title="Who plays which colour">
  <div class="player" id="pl-robot"><span class="face" id="fc-robot"></span><div><div class="nm" id="nm-robot">Robot</div><div class="st" id="st-robot"></div></div></div>
  <button class="swapbtn" id="swapbtn" title="Swap the colours" aria-label="Swap the colours">⇄</button>
  <div class="player" id="pl-human"><span class="face" id="fc-human"></span><div><div class="nm" id="nm-human">You</div><div class="st" id="st-human"></div></div></div>
 </div>
</header>
<section class="card stage" id="stage">
 <div>
  <div id="prompt"><span id="ptext">…</span><span class="spin"></span></div>
  <div id="asks"></div>
  <div id="coach" class="mute"></div>
 </div>
 <div class="dicecol" id="dice"></div>
</section>
<section class="pair">
 <div class="card"><h2>The game</h2>
  <div id="edge" title="Who is ahead" hidden><span id="edge-l"></span><div class="edgebar"><i id="edge-f"></i></div><span id="edge-r"></span></div>
  <div class="pic"><svg id="board" viewBox="-20 -20 450 460" preserveAspectRatio="none"></svg></div>
  <div id="info"></div><div id="log"></div></div>
 <div class="card"><h2>The camera</h2>
  <div class="pic"><img id="photo" alt="top view" style="display:none"></div>
  <div class="mute camnote" style="margin-top:10px">The board from below, the same way round as the drawing. Circles: where the game knows the checkers are.</div></div>
</section>
<section class="low">
  <details class="card"><summary class="mute">Original camera photo</summary>
   <img id="top" class="photo2" alt="raw photo"></details>
  <details class="card"><summary class="mute">Settings (used from the next New game)</summary>
   <div class="grid" id="settings" style="margin-top:8px"></div></details>
  <div id="warns"></div>
</section></main><div id="confetti"></div>
<script>
const OPTS={mode:["robot_vs_human","robot_vs_robot","watch"],engine:["gnubg","simple","random"],
 dice:["digital","typed","camera"],human_move:["done_button","robot_moves"],
 coach:[false,true],first:["human","robot","roll"],board:["auto","real","sim"]};
const NAMES={mode:"Mode",engine:"Engine",dice:"Dice",human_move:"Your move",coach:"Coach",first:"Who starts",board:"Board"};
const BTN={new:"New game",done:"I'm done",accept:"Accept",reread:"Read again",start:"Start",
 arrange:"Robot arranges from the trays",rearrange:"Robot arranges from the last game",go:"Yes, go",cancel:"Cancel",undo:"Undo last move",replay:"I put them back, play again",rolled:"Rolled"};
let S=null, built=false, lastPhoto=0, ED=null, PEND=null;
// -- the board is drawn as the corrected camera photo shows it: frame mm (x, y from the near side)
// -> magnet mm (x right, y down) through the layout's own map (s.frame), 1 unit = 1 mm --
// FLIP: the drawing is shown upside down compared with the camera's view (Nadav 14.11); the
// camera's picture is turned the same way by the game before it is sent
const FLIP=true;
function P(x,y){const A=S.frame,b=A[1][0]*x+A[1][1]*y+A[1][2];return [A[0][0]*x+A[0][1]*y+A[0][2],FLIP?420-b:b]}
function Q(x,y){const [a,b]=P(x,y);return `${a.toFixed(1)},${b.toFixed(1)}`}
function toFrame(mx,my){if(FLIP)my=420-my;const A=S.frame,d=A[0][0]*A[1][1]-A[0][1]*A[1][0],u=mx-A[0][2],v=my-A[1][2];
 return [(A[1][1]*u-A[0][1]*v)/d,(-A[1][0]*u+A[0][0]*v)/d]}
function txt(x,y,t,size,fill,bold){const [a,b]=P(x,y);
 return `<text x="${a}" y="${b+size*0.35}" font-size="${size}" ${bold?'font-weight="bold"':''} text-anchor="middle" fill="${fill}">${t}</text>`}
// the checkers are black and white, as on the real board (Nadav 15.11)
function isWhite(side){return (side==="robot")===!!(S&&S.robot_white)}
function ink(side){return isWhite(side)?"#1a1a1a":"#f4f4f4"}
function hex(side){return isWhite(side)?"#f2f2ea":"#1c1c1c"}
function disc(x,y,side){const [a,b]=P(x,y),w=isWhite(side);
 return `<circle cx="${a+1}" cy="${b+2}" r="12.6" fill="#000" opacity=".35"/>`+
  `<circle cx="${a}" cy="${b}" r="12.4" fill="url(#${w?'gW':'gB'})" stroke="${w?'#8d877a':'#000'}" stroke-width="1"/>`+
  `<circle cx="${a}" cy="${b}" r="8" fill="none" stroke="${w?'#b9b3a4':'#4d4d4d'}" stroke-width="1"/>`}
const DEFS=`<defs><radialGradient id="gW" cx="35%" cy="30%"><stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#d6d1c4"/></radialGradient>
 <radialGradient id="gB" cx="35%" cy="30%"><stop offset="0" stop-color="#5a5a5a"/><stop offset="1" stop-color="#0d0d0d"/></radialGradient>
 <linearGradient id="gWood" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#6b4427"/><stop offset="1" stop-color="#3a2414"/></linearGradient>
 <marker id="arr" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0 0L10 5L0 10z" fill="#ffe14d"/></marker></defs>`;
function base(s,edit){let h=DEFS+`<rect x="-20" y="-20" width="450" height="460" fill="url(#gWood)"/>`+
  `<rect x="-6" y="-6" width="422" height="432" rx="6" fill="#1d4d3f" stroke="${edit?'#ffc21a':'#2a1a0e'}" stroke-width="${edit?5:2}"/>`;
 if(edit)h+=`<rect x="-6" y="-6" width="422" height="432" rx="6" fill="#ffc21a" opacity=".10"/>`;
 const by=s.bar_y;h+=`<polygon points="${Q(0,by-14)} ${Q(410,by-14)} ${Q(410,by+14)} ${Q(0,by+14)}" fill="#000" opacity=".22"/>`;
 for(const [x,y,n] of s.labels){const left=x<205;
  const pts=left?`${Q(x-13,y+13)} ${Q(x-13,y-13)} ${Q(x+150,y)}`:`${Q(x+13,y+13)} ${Q(x+13,y-13)} ${Q(x-150,y)}`;
  h+=`<polygon points="${pts}" fill="${n%2?'#e3c28f':'#a8452e'}" opacity=".9"/>`;
  h+=txt(left?x+162:x-162,y,n,11,"#cfe3da")}
 return h}
function pathsSvg(s){let h="";const ps=s.paths||[];
 ps.forEach((p,i)=>{if(p.length<2)return;const last=i===ps.length-1;
  const pts=p.map(([x,y])=>`${x},${FLIP?420-y:y}`).join(" ");
  h+=`<polyline points="${pts}" fill="none" stroke="#ffe14d" stroke-width="${last?3.2:2}" stroke-linejoin="round" stroke-linecap="round" opacity="${last?1:.35}" ${last?'marker-end="url(#arr)"':''}/>`;
  if(last)h+=`<circle cx="${p[0][0]}" cy="${FLIP?420-p[0][1]:p[0][1]}" r="4" fill="#ffe14d"/>`});
 return h}
function edAdd(p,side){if(p===0||p===25){if(side==="robot")ED[0]++;else ED[25]--}
 else if(side==="robot"){if(ED[p]<0)ED[p]=1;else ED[p]++}else{if(ED[p]>0)ED[p]=-1;else ED[p]--}drawEdit(S)}
function edAsk(ev,p){// a new column (or the bar): which colour to add?
 let m=document.getElementById("edask");if(m)m.remove();
 m=el("div",{id:"edask",class:"edask"});m.style.left=Math.min(ev.clientX,innerWidth-190)+"px";m.style.top=ev.clientY+"px";
 m.append(el("div",{text:"Whose checker to add?"}));
 for(const side of ["robot","human"]){const b=el("button",{class:"sec",text:S.players[side].name+" ("+(isWhite(side)?"white":"black")+")",onclick:e=>{e.stopPropagation();m.remove();edAdd(p,side)}});
  b.style.borderLeft="14px solid "+hex(side);m.append(b)}
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
 let h=base(s,true)+txt(205,s.labels[11][1],"EDITING",13,"#ffc21a",true);
 for(const [x,y,n] of s.labels){const v=ED[25-n];if(v)h+=stack(x,y,x<205?1:-1,v,v>0?"robot":"human")}
 for(let i=0;i<ED[0];i++)h+=disc(205,s.bar_y+barStep(2*i)*BARP,"robot");
 for(let i=0;i<-ED[25];i++)h+=disc(205,s.bar_y+barStep(2*i+1)*BARP,"human");
 const o=offOf(ED);h+=txt(205,35,`off: ${s.players.robot.name} ${o[0]} · ${s.players.human.name} ${o[1]}`,12,"#cfe3da");
 svg.innerHTML=h}
function drawBoard(s){const svg=document.getElementById("board");
 let h=base(s,false);
 for(const [x,y,side] of s.pieces)h+=disc(x,y,side);
 svg.innerHTML=h+pathsSvg(s)}
// -- dice: always in the same place at the right, one under the other, one of them red --
const PIPS={1:[[2,2]],2:[[1,1],[3,3]],3:[[1,1],[2,2],[3,3]],4:[[1,1],[3,1],[1,3],[3,3]],
 5:[[1,1],[3,1],[2,2],[1,3],[3,3]],6:[[1,1],[3,1],[1,2],[3,2],[1,3],[3,3]]};
function face(v,red){return `<svg viewBox="0 0 40 40" width="100%" height="100%"><rect x="1" y="1" width="38" height="38" rx="8" fill="${red?'#c62828':'#fbf8ef'}" stroke="${red?'#7f1515':'#b9b2a0'}"/>`+
 `<rect x="3" y="3" width="34" height="14" rx="6" fill="#fff" opacity="${red?.12:.5}"/>`+
 (PIPS[v]||[]).map(([i,j])=>`<circle cx="${i*10}" cy="${j*10}" r="3.7" fill="${red?'#fff':'#1b1b1b'}"/>`).join("")+`</svg>`}
let lastDice, rollEnd=0, AC=null;
function rolling(){return Date.now()<rollEnd}
// two switches kept in this browser: sounds on / off, video mode (only the game on the page)
let SOUND=true, CLEAN=false;
try{SOUND=localStorage.getItem("sound")!=="0";CLEAN=localStorage.getItem("clean")==="1"}catch(e){}
function paintToggles(){document.getElementById("sndbtn").textContent=SOUND?"🔊":"🔇";
 document.getElementById("cleanbtn").textContent=CLEAN?"🎬 on":"🎬";document.body.classList.toggle("clean",CLEAN)}
function audio(){if(!SOUND)return null;try{AC=AC||new (window.AudioContext||window.webkitAudioContext)();if(AC.state==="suspended")AC.resume();return AC}catch(e){return null}}
function rattle(){const A=audio();if(!A)return;try{const t0=A.currentTime;
 for(let k=0;k<7;k++){const t=t0+k*0.07+Math.random()*0.03,n=A.createBufferSource(),b=A.createBuffer(1,A.sampleRate*0.03,A.sampleRate),
  d=b.getChannelData(0);for(let i=0;i<d.length;i++)d[i]=(Math.random()*2-1)*Math.pow(1-i/d.length,3);
  const f=A.createBiquadFilter();f.type="bandpass";f.frequency.value=1800+Math.random()*1800;f.Q.value=2;
  const g=A.createGain();g.gain.value=0.5*(1-k/9);n.buffer=b;n.connect(f).connect(g).connect(A.destination);n.start(t)}}catch(e){}}
function tone(A,type,f0,f1,t,dur,vol){const o=A.createOscillator(),g=A.createGain();o.type=type;
 o.frequency.setValueAtTime(f0,t);if(f1!==f0)o.frequency.exponentialRampToValueAtTime(f1,t+dur);
 g.gain.setValueAtTime(0.0001,t);g.gain.exponentialRampToValueAtTime(vol,t+0.02);g.gain.exponentialRampToValueAtTime(0.0001,t+dur);
 o.connect(g).connect(A.destination);o.start(t);o.stop(t+dur+0.05)}
function fanfare(){const A=audio();if(!A)return;const t=A.currentTime+0.05;   // a rising arpeggio and a held chord
 [523.25,659.25,783.99,1046.5].forEach((f,i)=>{tone(A,"triangle",f,f,t+i*0.13,0.22,0.3);tone(A,"square",f,f,t+i*0.13,0.18,0.05)});
 [523.25,659.25,783.99,1046.5,1318.5].forEach(f=>tone(A,"triangle",f,f,t+0.56,1.1,0.16))}
function yourTurn(){// a soft double chime (and a buzz on a phone): look up from the board, it is your move
 try{if(SOUND&&navigator.vibrate)navigator.vibrate([90,60,90])}catch(e){}
 const A=audio();if(!A)return;const t=A.currentTime+0.02;tone(A,"sine",880,880,t,0.18,0.22);tone(A,"sine",1318.5,1318.5,t+0.16,0.32,0.2)}
function sadTune(){const A=audio();if(!A)return;const t=A.currentTime+0.05;   // the falling "wah wah wah waaah"
 [[392,370,0,0.32],[370,349,0.36,0.32],[349,330,0.72,0.32],[330,247,1.08,1.0]].forEach(([a,b,d,len])=>{
  tone(A,"sawtooth",a,b,t+d,len,0.13);tone(A,"triangle",a/2,b/2,t+d,len,0.16)})}
function confetti(){const c=document.getElementById("confetti");c.innerHTML="";const cols=["#e2b053","#f6f1e4","#c62828","#3f9c82","#111"];
 for(let i=0;i<90;i++){const e=document.createElement("i");e.style.left=Math.random()*100+"vw";e.style.background=cols[i%cols.length];
  e.style.animationDuration=(2.2+Math.random()*2.4)+"s";e.style.animationDelay=Math.random()*1.2+"s";e.style.transform=`rotate(${Math.random()*360}deg)`;c.append(e)}
 setTimeout(()=>{c.innerHTML=""},6500)}
function roll(){rollEnd=Date.now()+700;rattle();const t=setInterval(()=>{const ds=document.querySelectorAll(".die[data-v]");
 for(const e of ds){const red=e.dataset.red==="1";if(rolling())e.innerHTML=face(1+Math.floor(Math.random()*6),red);else{e.innerHTML=face(+e.dataset.v,red);e.classList.remove("rolling")}}
 if(!rolling())clearInterval(t)},80)}
document.addEventListener("pointerdown",audio,{once:true});
// a click: the stage greys out and shows it is working at once, until the game answers
function stKey(s){return s?JSON.stringify([s.prompt,s.phase,s.asks,s.history.length]):""}
function post(a){if(a.name!=="settings"&&a.name!=="swap_colours"){PEND={t:Date.now(),seen:false,k:stKey(S)};document.getElementById("stage").classList.add("busy")}
 return fetch("/action",{method:"POST",body:JSON.stringify(a)})}
function el(t,attrs={},kids=[]){const e=document.createElement(t);for(const k in attrs){
 if(k=="text")e.textContent=attrs[k];else if(k.startsWith("on"))e[k]=attrs[k];else e.setAttribute(k,attrs[k])}
 for(const c of kids)e.append(c);return e}
function buildSettings(s){const g=document.getElementById("settings");g.innerHTML="";
 for(const k in OPTS){const sel=el("select",{onchange:()=>{let v=sel.value;if(v==="true")v=true;if(v==="false")v=false;
   post({name:"settings",values:{[k]:v}})}});
  for(const o of OPTS[k]){const op=el("option",{value:String(o),text:String(o).replaceAll("_"," ")});
   if(String(s[k])===String(o))op.selected=true;sel.append(op)}
  g.append(el("label",{text:NAMES[k]}),sel)}}
// -- who plays what: a computer and a person, drawn in the colour of their checkers --
function icon(kind,white){const f=white?"#f6f4ec":"#141414",o=white?"#141414":"#f6f4ec";
 const st=`fill="${f}" stroke="${o}" stroke-width="1.3" stroke-linejoin="round"`;
 return kind==="computer"
  ?`<svg viewBox="0 0 32 32"><rect x="3" y="5" width="26" height="17" rx="2.5" ${st}/><rect x="6.5" y="8.5" width="19" height="10" rx="1" fill="${o}" opacity=".85"/>`+
   `<circle cx="12.5" cy="13.5" r="1.7" fill="${f}"/><circle cx="19.5" cy="13.5" r="1.7" fill="${f}"/><path d="M13 22h6l1.2 4H11.8z" ${st}/><rect x="8" y="26" width="16" height="2.6" rx="1.3" ${st}/></svg>`
  :`<svg viewBox="0 0 32 32"><circle cx="16" cy="10" r="6" ${st}/><path d="M4.5 28.5c0-7 5-10.5 11.5-10.5s11.5 3.500 11.500 10.500z" ${st}/></svg>`}
function paintPlayers(s){for(const side of ["robot","human"]){const p=s.players[side],i=side==="robot"?0:1;
  const fc=document.getElementById("fc-"+side),k=p.kind+isWhite(side);if(fc.dataset.k!==k){fc.dataset.k=k;fc.innerHTML=icon(p.kind,isWhite(side))}
  document.getElementById("nm-"+side).textContent=p.name;
  document.getElementById("st-"+side).textContent=(isWhite(side)?"white":"black")+" · "+s.pips[i]+" to go"+(s.off[i]?" · "+s.off[i]+" off":"");
  document.getElementById("pl-"+side).classList.toggle("on",!s.winner&&s.turn===side||!!s.winner&&s.winner.who===side)}}
document.getElementById("swapbtn").onclick=()=>{if(!S)return;S.robot_white=!S.robot_white;post({name:"swap_colours"});
 paintPlayers(S);if(ED)drawEdit(S);else drawBoard(S)};
let newArmed=0, lastWin=null, lastMine=false;
function newGame(b){if(S&&S.turn&&!S.winner&&Date.now()>newArmed){newArmed=Date.now()+3000;b.textContent="Stop this game? Click again";
  setTimeout(()=>{b.textContent="New game"},3000);return}
 newArmed=0;post({name:"new"})}
function drawDice(s){const box=document.getElementById("dice"),d=s.dice||[0,0],k=JSON.stringify(d);
 if(box.dataset.k===k&&!rolling())return;box.dataset.k=k;box.innerHTML="";
 d.forEach((v,i)=>{const e=el("span",{class:"die"+(v?(rolling()?" rolling":""):" none")});
  if(v){e.dataset.v=v;e.dataset.red=i?"1":"0"}e.innerHTML=face(v&&rolling()?1+Math.floor(Math.random()*6):v,i===1);box.append(e)})}
function render(s){S=s;if(!built){buildSettings(s.settings);built=true}
 document.getElementById("sim").textContent=s.sim?"simulator":"";
 // busy: right after a click, and for as long as the game works without waiting for one
 if(PEND){if(s.busy)PEND.seen=true;if((PEND.seen&&!s.busy)||(!s.busy&&stKey(s)!==PEND.k)||(!PEND.seen&&Date.now()-PEND.t>1500))PEND=null}
 const st=document.getElementById("stage");st.classList.toggle("busy",!!(s.busy||PEND));
 st.classList.toggle("won",!!s.winner&&!s.winner.lost);st.classList.toggle("lost",!!s.winner&&!!s.winner.lost);
 const wk=s.winner?JSON.stringify(s.winner)+s.history.length:null;
 if(wk&&wk!==lastWin&&lastWin!==undefined){if(s.winner.lost)sadTune();else{fanfare();confetti()}}
 lastWin=wk;
 const mine=s.phase==="human_move"&&s.asks.includes("done")&&!s.sim;   // a person's move starts: chime
 if(mine&&!lastMine&&s.history.length)yourTurn();lastMine=mine;
 document.getElementById("ptext").textContent=s.prompt;
 const dk=JSON.stringify(s.dice||null);if(dk!==lastDice){if(lastDice!==undefined&&s.dice)roll();lastDice=dk}
 drawDice(s);
 const a=document.getElementById("asks");const keep=document.activeElement&&a.contains(document.activeElement)&&document.activeElement.tagName!=="BUTTON";
 const key=JSON.stringify([s.asks,s.legal.length,s.likely,!!s.winner,s.phase]);
 if(a.dataset.k!==key&&!(keep&&a.dataset.a===JSON.stringify(s.asks))){a.innerHTML="";a.dataset.k=key;a.dataset.a=JSON.stringify(s.asks);
  for(const k of s.asks){
   if(k==="dice_input"){const fix=s.asks.includes("done"),i=el("input",{placeholder:fix?"wrong dice? e.g. 5 2":"dice, e.g. 5 2",size:fix?15:8});
    i.onkeydown=ev=>{if(ev.key==="Enter"){ev.preventDefault();post({name:"dice",value:i.value})}};
    a.append(i,el("button",{class:fix?"sec":"",text:fix?"Fix dice":"OK",onclick:()=>post({name:"dice",value:i.value})}))}
   else if(k==="likely"){for(const m of s.likely)a.append(el("button",{text:m,onclick:()=>post({name:"move",value:m})}))}
   else if(k==="edit"){
    a.append(el("div",{class:"hint",text:"Or fix the drawing of the board: click a checker to take it off, the empty part of a column to add one, an empty column or the bar to choose whose."}),
     el("button",{class:"sec",text:"Use the board as drawn",onclick:()=>post({name:"board",value:ED})}),
     el("button",{class:"sec",text:"Undo my changes",onclick:()=>{ED=S.seen.slice();drawEdit(S)}}))}
   else if(k==="move_pick"){const sel=el("select");for(const m of s.legal)sel.append(el("option",{text:m}));
    a.append(sel,el("button",{class:s.asks.includes("likely")?"sec":"",text:s.asks.includes("likely")?"It was this one":"Play this",onclick:()=>post({name:"move",value:sel.value})}))}
   else if(k==="undo"){const ub=el("button",{class:"sec",text:BTN.undo});   // the robot moves: ask twice
    ub.onclick=()=>{if(ub.dataset.armed){post({name:"undo"})}else{ub.dataset.armed="1";ub.textContent="Undo your move and the robot's? Click again";
     setTimeout(()=>{delete ub.dataset.armed;ub.textContent=BTN.undo},3000)}};a.append(ub)}
   else if(k==="new")a.append(el("button",{class:"big",text:"New game",onclick:()=>post({name:"new"})}));
   else a.append(el("button",{class:(k==="done"||k==="start"||k==="rolled")?"big":(k==="reread"||k==="arrange"||k==="rearrange"||k==="cancel"||k==="replay")?"sec":"",text:BTN[k]||k,onclick:()=>post({name:k})}))}
  if(s.phase==="idle")a.append(el("button",{class:"big",text:"New game",onclick:()=>post({name:"new"})}))}
 document.getElementById("newbtn").hidden=s.phase==="idle"||s.asks.includes("new");
 paintPlayers(s);
 const eg=document.getElementById("edge");eg.hidden=!s.edge;
 if(s.edge){const pr=Math.round(s.edge.robot*100),bar=eg.querySelector(".edgebar"),f=document.getElementById("edge-f");
  // the robot's share from the left in the robot's checker colour, the rest in the other colour
  f.style.width=pr+"%";f.style.background=hex("robot");bar.style.background=hex("human");
  document.getElementById("edge-l").textContent=s.players.robot.name+" "+pr+"%";
  document.getElementById("edge-r").textContent=(100-pr)+"% "+s.players.human.name;
  eg.title="Chance to win, by "+s.edge.by+(s.edge.by==="pip count"?" (the race only: hits and blocks are not counted)":"")}
 document.getElementById("coach").textContent=s.coach||"";
 document.getElementById("info").textContent=s.info||"";
 const lg=document.getElementById("log");lg.innerHTML="";
 for(const m of s.history.slice(-5).reverse())lg.append(el("div",{},[el("b",{text:m.n+". "+((s.players[m.who]||{}).name||m.who)}),
  document.createTextNode(" "+m.dice.join("-")+":  "+(m.note==="no legal move"?"":(m.shown||m.move))+(m.note==="no legal move"?"no legal move":""))]));
 const w=document.getElementById("warns");w.innerHTML="";for(const t of s.warnings)w.append(el("div",{class:"warn",text:"⚠ "+t}));
 if(s.asks.includes("edit")&&s.seen){if(!ED)ED=s.seen.slice();drawEdit(s)}else{ED=null;drawBoard(s)}
 if(s.has_photo&&s.photo_t!==lastPhoto){lastPhoto=s.photo_t;const im=document.getElementById("photo");
  im.src="/top.jpg?t="+s.photo_t;im.style.display="block";
  const tp=document.getElementById("top");tp.src="/photo.jpg?t="+s.photo_t;tp.style.display="block"}}
async function tick(){try{render(await (await fetch("/state")).json())}catch(e){}setTimeout(tick,300)}
document.getElementById("board").addEventListener("click",edPlace);
document.getElementById("sndbtn").onclick=()=>{SOUND=!SOUND;try{localStorage.setItem("sound",SOUND?"1":"0")}catch(e){}paintToggles();if(SOUND)yourTurn()};
document.getElementById("cleanbtn").onclick=()=>{CLEAN=!CLEAN;try{localStorage.setItem("clean",CLEAN?"1":"0")}catch(e){}paintToggles()};
paintToggles();
document.getElementById("newbtn").onclick=ev=>newGame(ev.currentTarget);
document.addEventListener("keydown",ev=>{// Enter = I'm done, when it is offered and nothing is being typed
 if(ev.key!=="Enter"||!S||!S.asks.includes("done")||S.busy||PEND)return;
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
    keys = {"n": "new", "s": "start", "d": "done", "a": "accept", "r": "reread", "g": "arrange", "o": "rolled"}
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
            print("n=new s=start g=arrange d=done r=read again a=accept o=rolled 52=dice m <move> q=quit")


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
    print("Terminal: n=new game s=start d=done 52=dice q=quit")
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
