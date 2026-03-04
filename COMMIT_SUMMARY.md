# Recent repository changes

This document summarises the contents of the two most recent commits on the `main` branch and offers some ideas for follow‑up improvements.

---

## 1. `fd5916a5…` – *"new gradio interface"* (2024‑03‑29)

### What was added/modified

- **User interface overhaul**: complete rewrite of `latentblending/gradio_ui.py`.
  - Introduces a `MultiUserRouter` that holds a dictionary of `BlendingEngine` instances for different models and handles per‑user state via `BlendingVariableHolder`.
  - Adds controls for prompt/negative‑prompt, seed management, preview frames, movie sequencing, deletion/reordering and final video generation.
  - Drops earlier in‑paint UI code, simplifies workflow to keyframe‑→‑movie.
  - Adds command‑line flags (`--do_compile`, `--nmb_preview_images`, `--server_name`).
  - Thorough cleanup of warnings and performance tweaks (e.g. `torch.backends.cudnn.benchmark = False`).
- **Documentation update**: small change in `README.md` (added/adjusted Gradio instructions).
- A large number of lines were inserted and some blocks removed to support the new interface
  (152 insertions, 71 deletions according to `git show --stat`).

### Impact

This commit makes it much easier for users to start a session, pick a model/resolution, generate and select preview images, and then assemble a transition movie from several keyframes all within a browser.  The architecture also makes it straightforward to later extend the UI to multiple concurrent sessions or to hook it up to a HuggingFace Space.

---

## 2. `27681107…` – *"draft for a blend from images option"* (2026‑01‑19)

### What was added

- New CLI tools that work with pre‑computed VAE latents:
  - `preprocess_images.py` – load two input images, resize (optional fixed size or round to 32 multiples), encode them with the pipeline's VAE and dump the latents to disk, plus produce a `preprocess_manifest.json` with metadata.  It also offers an `--ask-to-blend` flag prompting the user to continue.
  - `blend_from_images.py` – read a manifest produced by the previous script, load the two latents, interpolate between them (spherical or linear) to produce a sequence of frames and optionally stitch them into an MP4.
- Both scripts are fully argument‑driven, perform basic validation (existence, shape matching, multiple‑of‑32 sizes) and respect CUDA availability.
- No changes to existing library modules; the new functionality lives in standalone scripts.

### Impact

These tools allow the same latent‑blending machinery to operate on user‑supplied photographs rather than prompts, vastly expanding the possible use cases (e.g. morphing between two real pictures).  The manifest format makes it easy to connect with downstream processes and keeps the pipeline reproducible.

---

# Suggestions & next steps

Below are some ideas to consolidate the recent work, increase usability, and open new directions.

1. **Script accorpamento (consolidation)**
   - Merge `preprocess_images.py` and `blend_from_images.py` into a single CLI (`preprocess_images.py`) with subcommands (`preprocess`, `blend`, `all`) and deprecate the original `blend_from_images.py` via a thin wrapper.
   - Add an `--interactive` mode that, after preprocessing, immediately proceeds with blending using the same CLI session.
   - Consider a `--from-latents` flag to the CLI so that it can also accept existing `.pt` files (skip manifest generation).
   - Update documentation and examples accordingly (see README section added above).   - **Next tasks**:
     * investigate a macOS diffusers pipeline as requested by the user.
     * explore a new "folder-to-blend" feature where images from a directory
       are processed alphabetically to create multi-step transitions.
2. **Gradio UI enhancements**
   - Provide an “upload images” tab that uses the new image‑preprocessing scripts behind the scenes, allowing users to morph between photos in the browser.
   - Add sliders to control the interpolation type and number of frames when blending from images.
   - Expose the new CLI parameters (size, model override, fps, mp4 path) via GUI controls.
   - Investigate caching pipelines per user/model to reduce load time on repeated launches.

3. **Documentation & examples**
   - Update the README with explicit examples demonstrating the new image‑based workflow (`pipelines/images_to_transition.py --image1 foo.jpg --image2 bar.png ...`).
   - Add a short tutorial or notebook that starts from two JPGs and produces a video, possibly including before/after screenshots.
   - Maintain a `CHANGELOG.md` if you want to track future commits systematically.

4. **New perspectives / extensions**
   - Implement multi‑image blending (morph an arbitrary sequence of input photographs) and add a corresponding JSON manifest format describing all of them.
   - Add audio‑sync support: allow specifying an audio file and automatically adjusting frame counts to match beats or offsets.
   - Expose the latent‑interpolation utilities (`_load_latent`, `interpolate_spherical/linear`) in the public API for users building custom scripts.
   - Provide hooks for ControlNet, IP‑Adapter or other aux models (already mentioned for the future).  The Gradio UI could let users toggle those on/off per segment.
   - Offer a “headless server” mode where the CLI listens for requests (e.g. via REST or websocket) to generate blends on demand; useful for integration into a larger pipeline or webapp.

5. **Packaging and distribution**
   - Ensure `preprocess_images.py` and `blend_from_images.py` are installed as console scripts in `setup.py` so they become available after `pip install git+...`.
   - Add tests for the new CLI routines to catch regressions (the current repo seems light on tests).

6. **Performance / technical tweaks**
   - Allow `blend_from_images.py` to run without launching the full diffusers pipeline when only interpolation is needed (i.e. use pre‑loaded latents exclusively).
   - Add support for batching multiple pairs in one invocation (accorpamento of latents in a single run).
   - Provide a utility to visualise the manifest structure or convert between manifest versions if you evolve it.

---

Concluding note: the last two commits have dramatically expanded the project’s capabilities both from a user‑interface and from a data‑input perspective.  Turning the scripts into a more unified, user‑friendly CLI and continuing to invest in the Gradio interface (especially for photo morphing) will make the repository accessible to a much wider audience.

Feel free to tweak or add to this document as the project evolves.

---

## Delta update (2026-03-04)

- Fixed an incomplete/corrupted `images_to_transition.py` by replacing it with a stable compatibility wrapper.
- Applied automatic device routing for macOS pipeline support in active flows (`cuda -> mps -> cpu`) via `latentblending.utils.get_device()`.
- Enabled this routing in both `preprocess_images.py` and `latentblending/gradio_ui.py`.