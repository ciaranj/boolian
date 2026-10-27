import type { Action, Emotion } from "./ghost/ghost";

export interface PuppetLine {
  emotion: Emotion;
  action: Action;
  text: string;
}

/** Number keys 1-9. Spoken through the server's TTS, or the browser's voice if offline. */
export const PUPPET_LINES: PuppetLine[] = [
  { emotion: "happy", action: "bounce", text: "Happy Halloween! Oh, I do love your costume!" },
  { emotion: "mischievous", action: "peek", text: "Boo! ...Oh dear, did that work? I've been practising for a hundred and seventy-three years." },
  { emotion: "happy", action: "tilt", text: "Trick or treat? I'd offer you a sweet, but my hands go straight through the bowl." },
  { emotion: "curious", action: "tilt", text: "Ooh, visitors! Come closer, I don't bite. I don't even have teeth!" },
  { emotion: "mischievous", action: "hide", text: "Shhh. Can you hear that? No? Me neither. Spooky!" },
  { emotion: "happy", action: "spin", text: "Wheee! Watch this!" },
  { emotion: "confused", action: "tilt", text: "Pardon? My ears are a bit foggy tonight. Could you say that again?" },
  { emotion: "curious", action: "none", text: "Hmm, let me think about that. Ghosts think very, very slowly." },
  { emotion: "happy", action: "bounce", text: "Goodbye! Come back and haunt me again soon!" },
  { emotion: "mischievous", action: "peek", text: "Ready for my scariest noise? [raspberry] Ha! Got you!" },
  { emotion: "mischievous", action: "bounce", text: "Oops! [fart] Excuse me! That was a bat, honestly." },
];
