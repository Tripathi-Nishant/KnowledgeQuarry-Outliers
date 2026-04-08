# Video Scripts: Bottleneck RL Project

---

## Script A: The 60-Second "Elevator Pitch"
*(Goal: Maximum Impact. Fast-paced.)*

| Time | Visual Action | Voiceover (The Script) |
| :--- | :--- | :--- |
| **0:00** | **Overlay**: "The Bottleneck Problem". | "Urban traffic isn't just about volume. It’s about behavior." |
| **0:10** | **Show**: Pygame jam with many RED cars. | "Aggressive driving at bottlenecks creates a cascade of braking that collapses throughput by 40%. We call this the behavioral tipping point." |
| **0:25** | **Show**: GREEN cars merging smoothly. | "We used Reinforcement Learning to train cooperative agents. They don't just move fast—they coordinate." |
| **0:40** | **Show**: Comparison plots (6% Gain). | "The result? Adding just 20% AI agents to a mixed population increases overall throughput by 6% and system speed by 7.5%." |
| **0:50** | **Show**: Your Name/Title. | "Solving gridlock with intelligence, not infrastructure. That’s the future of the smart city." |

---

## Script B: The 120-Second "Deep Dive"
*(Goal: Technical Depth. Educational.)*

| Time | Visual Action | Voiceover (The Script) |
| :--- | :--- | :--- |
| **0:00** | **Show**: Road Layout (Empty). | "Bottlenecks, like lane merges, are the primary source of urban congestion. In this project, we explored how decentralized behaviors lead to emergent gridlock." |
| **0:20** | **Show**: Tipping Point plot. | "We discovered a phase transition: once 30% of drivers act selfishly, the road collapses. This proves congestion is a behavioral coordination failure." |
| **0:40** | **Show**: Code Snippet (12-Sensor Suite). | "To solve this, we built a custom Gymnasium environment. We trained agents using PPO—Proximal Policy Optimization—with a 12-sensor suite that looks at lane occupancy and speed gradients." |
| **1:05** | **Show**: Training Curriculum Chart. | "Our agents learned through a curriculum—starting in easy traffic and mastering high-density chaos. They were rewarded for their own speed but penalized for stopping the collective flow." |
| **1:30** | **Show**: Live Pygame Comparison. | "Notice how the Green agents merge early and create gaps. This 'flow-dampening' behavior stabilizes the road, achieving a 6% throughput gain with only a 20% deployment." |
| **1:50** | **Show**: Conclusion Slide. | "This framework proves that behavioral AI can significantly increase road capacity without expanding physical infrastructure. Thank you for watching." |

---

## Tips for Recording:
1.  **Screen Recording**: Use OBS or Windows Game Bar (`Win+G`) to record your screen while running `pygame_renderer.py`.
2.  **Toggle Aggression**: While recording, press `D` to increase aggression to show the "Traffic Collapse," then let the RL agents (Green) clean it up.
3.  **Use the Notebook**: Briefly show the `analysis.ipynb` notebook to prove you have a full research pipeline.
