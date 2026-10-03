// 3D 数字人：TalkingHead（three.js 渲染 3D 人物）+ HeadAudio（听声音算口型，不需要文字和时间戳，中文也能用）。
// 两个库都是 MIT 许可，从 jsDelivr 加载（页面要先写好 importmap：three / three/addons/ / talkinghead）。
//
//   const h = await createHead(容器, AudioContext, 声音总线(AudioNode), {url, body, onprogress});
//   声音总线上过的声音 → 自动出口型；h.setMood("happy"); h.dispose();
//
// 人物模型要求：Mixamo 骨骼 + ARKit 52 个表情 + Oculus 15 个口型 blendshape（Avaturn / VRoid 转换 / MPFB 导出都行）。
import { TalkingHead } from "talkinghead";
import { HeadAudio } from "https://cdn.jsdelivr.net/gh/met4citizen/HeadAudio@0.1.0-alpha/dist/headaudio.min.mjs";

const HA = "https://cdn.jsdelivr.net/gh/met4citizen/HeadAudio@0.1.0-alpha/dist/";
const GH = "https://cdn.jsdelivr.net/gh/met4citizen/TalkingHead@main/avatars/";

// 示例模型（TalkingHead 仓库自带）：只能非商用——测试、演示用；正式上线换成自己做的模型。
// （vroid.glb 用了 meshopt 压缩，TalkingHead 1.7 的加载器打不开，不放进来。）
export const PRESETS = {
  brunette: { url: GH + "brunette.glb", body: "F", name: "写实风女性（示例，非商用）", mb: 4.7 },
  mpfb: { url: "https://raw.githubusercontent.com/met4citizen/TalkingHead/main/avatars/mpfb.glb", body: "F",
          name: "MakeHuman 女性（CC0，可商用，但 37 MB 很大）", mb: 36.8 },
  avaturn: { url: GH + "avaturn.glb", body: "F", name: "照片生成风女性（示例，非商用）", mb: 13.8 },
  avatarsdk: { url: GH + "avatarsdk.glb", body: "M", name: "照片生成风男性（示例，非商用，T 恤上有厂商字样）", mb: 12.3 },
};

// 老师打的情绪标记（[[情绪：对方=…，我=…]] 里「我」的那个词）→ 3D 人物的表情
const MOODS = [
  [/开心|高兴|兴奋|好笑|轻松|俏皮|得意|欣慰|愉快|调皮|乐|笑/, "happy"],
  [/心疼|难过|伤心|难受|失落|沉重|遗憾|委屈|低落/, "sad"],
  [/担心|紧张|害怕|着急|不安|慌/, "fear"],
  [/生气|气愤|不爽|恼|火大|无语/, "angry"],
  [/温柔|喜欢|爱|甜|暖|疼惜|宠|感动/, "love"],
];
export function moodOf(text) {
  for (const [re, m] of MOODS) if (re.test(text || "")) return m;
  return "neutral";
}

export function resolveModel(spec, gender) {
  // spec：预设名 / 上传的 .glb 地址 / "" = 按性别选默认 / "none" = 不用 3D
  if (spec === "none") return null;
  if (spec && PRESETS[spec]) return PRESETS[spec];
  if (spec && /\.glb(\?|$)/i.test(spec)) return { url: spec, body: gender === "男" ? "M" : "F", name: "自己的模型" };
  return gender === "男" ? PRESETS.avatarsdk : PRESETS.brunette;
}

export async function createHead(node, ctx, input, { url, body = "F", onprogress = null, view = "upper" } = {}) {
  const head = new TalkingHead(node, {
    ttsEndpoint: null, lipsyncModules: ["en"], lipsyncLang: "en", audioCtx: ctx,
    cameraView: view, cameraRotateEnable: false, cameraZoomEnable: false, cameraPanEnable: false,
    modelPixelRatio: Math.min(2, window.devicePixelRatio || 1), modelFPS: 30,
    avatarIdleEyeContact: 0.6, avatarSpeakingEyeContact: 0.85, avatarIdleHeadMove: 0.5, avatarSpeakingHeadMove: 0.6,
    lightAmbientIntensity: 2.2,
  });
  await head.showAvatar({ url, body, avatarMood: "neutral", lipsyncLang: "en" }, onprogress);
  await ctx.audioWorklet.addModule(HA + "headworklet.min.mjs");
  const ha = new HeadAudio(ctx, { processorOptions: {}, parameterData: { vadGateActiveDb: -40, vadGateInactiveDb: -60 } });
  await ha.loadModel(HA + "model-en-mixed.bin");
  input.connect(ha);
  ha.onvalue = (key, value) => {
    const m = head.mtAvatar[key];
    if (m) Object.assign(m, { newvalue: value, needsUpdate: true });
  };
  head.opt.update = ha.update.bind(ha);
  let mood = "neutral";
  return {
    head, ha,
    setMood(m) {
      if (!m || m === mood) return;
      mood = m;
      try { head.setMood(m); } catch (e) { /* 模型没有这个表情就算了 */ }
    },
    gesture(name) { try { head.playGesture(name, 2); } catch (e) {} },
    dispose() {
      try { head.opt.update = null; } catch (e) {}
      try { input.disconnect(ha); } catch (e) {}
      try { head.stop(); } catch (e) {}
      node.innerHTML = "";
    },
  };
}
