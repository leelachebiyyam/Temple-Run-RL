# Temple-Run-RL — short demo script

Speak normally; this is roughly a minute to a minute and a half. The gameplay recording is silent so you can add your own voice.

**Show the game:**

“Hi, this is Temple-Run-RL. I wanted to see if I could teach an AI to survive Temple Run 2 instead of just giving it a fixed set of rules.

What you’re seeing here is the agent controlling the runner, not me. It can turn left or right, jump, slide, or do nothing.

**Keep the gameplay playing:**

I rebuilt the core gameplay in C and connected it to PufferLib. I analyzed the Android game to understand its movement, track pieces, and obstacles, and used extracted models and animations for the visuals.

The agent gets rewarded for clearing obstacles and collecting coins. If it dies, it loses reward. I’m not penalizing it just for swiping a lot—the priority is staying alive.

**Show the training terminal, or keep the gameplay playing:**

Training runs locally on my Mac using Apple Metal. The hardest part was getting the environment right. If a jump is impossible because of a collision bug, more training won’t fix it.

It’s still learning, and you’ll see it make mistakes. It’s not an exact copy of the game, but the environment, training, and agent-controlled playback are working.”
