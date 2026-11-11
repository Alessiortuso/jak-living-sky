# The merchant's JET-Board fight: The Forgotten Lands' PDS sentinel in Misty's arena

The boss of Kuitar5's *The Forgotten Lands* (github.com/Kuitar5/the-forgotten-lands, used with the author's
permission, credited in the README), fought in the arena at Misty Island: the merchant's third arena fight, which
puts the JET-Board on sale.

## How

- **A small extra level, `pdsarena`** (custom level, like Brume), loaded beside Misty while Jak is at the arena. No
  visible geometry of its own (a tiny mesh far below, so Jak is never "inside" it: Misty stays the level he's in,
  its look, sky and sounds). It holds only the fight's actors, as entities, so the mod's code reads them as it does
  in its own level (paths, gem order, linked traps and platforms, the custom nav mesh).
- **Positions**: the mod's arena (valley, centre 2075.05 7.93 1852.18 m) moved onto Misty's arena floor (centre
  -197.28, 28.0, 958.28 m; flat at about 28 m from 12 to 51 m out, four structures on the diagonals 39-48 m out),
  turned so the outer traps miss those structures.
- **Code**: the mod's boss files (pds-boss-setup, -states, -part, -extras) in `goal_src/jak1/levels/pdsarena/`,
  in the level's DGO. Their nav mesh `*nav-valley-boss*` moved with them. The mod's own helpers they use (macros,
  tfl-arcing-shot...) brought along or replaced by ours.
- **Enemies**: Misty's own (babak, bonelurker), not the mod's swamp rats, frogs, ice cubes, robbers: Misty gets no
  heavier. Their custom frog and ice-cube code dropped.
- **Dropped**: the lab door and the platforms after the fight (they lead into the mod's lab), the developer
  commentary nodes.
- **Models**: the mod's glbs (pds-sentinel, gems, traps, platforms, swing poles, camera) in custom_assets, built
  with build-actor, in the level's FR3 (its custom_models).
- **Joining the merchant's arena**: his menu starts the fight (the level loaded, the fight's master told to go);
  winning sets the third fight won (*arena-won* bit 2): the JET-Board on sale.

## Steps

1. The level: jsonc (actors moved), gd, an empty-ish glb, game.gp entries, level-info entry, models.
2. The code ported, compiling.
3. Loading it at the arena; the fight started from the merchant's menu.
4. Winning and losing; the reward; the credits.
