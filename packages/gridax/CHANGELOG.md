# Changelog

## [0.4.0](https://github.com/jejjohnson/gridax/compare/gridax-v0.3.5...gridax-v0.4.0) (2026-11-28)


### ⚠ BREAKING CHANGES

* **solvers:** drop the tridiagonal re-export of gaussx, and run the capacitance example ([#592](https://github.com/jejjohnson/gridax/issues/592))

### Bug Fixes

* **mask:** a periodic axis reads the mask's ghost ring wrapped ([#591](https://github.com/jejjohnson/gridax/issues/591)) ([173b331](https://github.com/jejjohnson/gridax/commit/173b3311f15cc1fe71a23ccdeedc7e17d51d3e64))


### Performance Improvements

* **advection:** call each tier once, so the gradient of wide WENO compiles in seconds ([#594](https://github.com/jejjohnson/gridax/issues/594)) ([aa19982](https://github.com/jejjohnson/gridax/commit/aa19982f514c6bb868766f3edbc15bb890104641))


### Code Refactoring

* **solvers:** drop the tridiagonal re-export of gaussx, and run the capacitance example ([#592](https://github.com/jejjohnson/gridax/issues/592)) ([29ddc70](https://github.com/jejjohnson/gridax/commit/29ddc701bc163b0b959dbd0d67c601b2a7022e35))

## [0.3.5](https://github.com/jejjohnson/gridax/compare/gridax-v0.3.4...gridax-v0.3.5) (2026-11-26)


### Bug Fixes

* **deps:** gridax asks for pipekit-ops 0.4.0, whose core it imports ([#588](https://github.com/jejjohnson/gridax/issues/588)) ([678b194](https://github.com/jejjohnson/gridax/commit/678b19412f49c4e21904b09cf55712b2091e9bdf))

## [0.3.4](https://github.com/jejjohnson/gridax/compare/gridax-v0.3.3...gridax-v0.3.4) (2026-11-26)


### Bug Fixes

* **deps:** pipekit at pipekit-array-v0.3.0 and geophysix v0.0.10 ([#586](https://github.com/jejjohnson/gridax/issues/586)) ([c1bc2d1](https://github.com/jejjohnson/gridax/commit/c1bc2d19c2555682a516d26dd324ca0fb6273954))

## [0.3.3](https://github.com/jejjohnson/gridax/compare/gridax-v0.3.2...gridax-v0.3.3) (2026-11-25)


### Bug Fixes

* **deps:** pipekit-array 0.2.0 and its areas ([#581](https://github.com/jejjohnson/gridax/issues/581)) ([5dd26f4](https://github.com/jejjohnson/gridax/commit/5dd26f49fba1a3249c4f66be58445635ff4f6e1c))

## [0.3.2](https://github.com/jejjohnson/gridax/compare/gridax-v0.3.1...gridax-v0.3.2) (2026-11-25)


### Bug Fixes

* **deps:** pipekit-array 0.1.6 and geophysix 0.0.8, the tags of the stack ([#574](https://github.com/jejjohnson/gridax/issues/574)) ([8b27fbb](https://github.com/jejjohnson/gridax/commit/8b27fbb9a0b0219a0c9819d7fd05b773fb114b8e))

## [0.3.1](https://github.com/jejjohnson/gridax/compare/gridax-v0.3.0...gridax-v0.3.1) (2026-11-24)


### Features

* **diagnostics:** strain, Okubo–Weiss, geostrophic velocity and potential vorticity on collocated grids ([#570](https://github.com/jejjohnson/gridax/issues/570)) ([9a9cb75](https://github.com/jejjohnson/gridax/commit/9a9cb756e049460e58472fabfb250392a2ccb23b))


### Bug Fixes

* **deps:** geophysix 0.0.7 in place of 0.0.3, so that a project can use both ([#571](https://github.com/jejjohnson/gridax/issues/571)) ([bfbdf57](https://github.com/jejjohnson/gridax/commit/bfbdf57e49ca417c46b88dbf24562cf5cbffa782))
* **fields:** pipekit's diff and coarsen on a CField no longer claim a point the values are not on ([#567](https://github.com/jejjohnson/gridax/issues/567)) ([0ee1c31](https://github.com/jejjohnson/gridax/commit/0ee1c31f019887a2603b5030fdcd8ee993ac7ca5))

## [0.3.0](https://github.com/jejjohnson/gridax/compare/gridax-v0.2.0...gridax-v0.3.0) (2026-11-23)


### ⚠ BREAKING CHANGES

* **mask:** one builder for the masks of every rank, and reconstruct_fn goes ([#557](https://github.com/jejjohnson/gridax/issues/557))
* **operators:** no operation with a mask reads land ([#554](https://github.com/jejjohnson/gridax/issues/554))
* **operators:** the collocated spherical laplacian is the flux form ([#549](https://github.com/jejjohnson/gridax/issues/549))
* removed, with the replacement after the colon (a function of the grid, or the method of GridOps(grid, mask)). Difference1D/2D/3D and SphericalDifference2D/3D: difference, divergence, curl, laplacian, grad_perp(at="T"), geostrophic_velocity. Interpolation1D/2D/3D: interpolate. Divergence2D and SphericalDivergence2D/3D: divergence. SphericalLaplacian2D/3D: laplacian. Vorticity2D/3D and SphericalVorticity2D/3D: curl, sw_potential_vorticity, pv_flux, enstrophy, potential_enstrophy. pv_flux_blend and pv_flux_arakawa_lamb: pv_flux(form="blend"). ArakawaJacobian2D: arakawa_jacobian. Coriolis2D/3D: coriolis. Energetics2D: kinetic_energy, bernoulli_potential, available_potential_energy. Strain2D: shear_strain, tensor_strain, strain_rate_squared, okubo_weiss_parameter. QGPotentialVorticity2D: qg_potential_vorticity, stretching_term. potential_vorticity_multilayer: qg_potential_vorticity(A=). sw_potential_vorticity_multilayer: sw_potential_vorticity. Advection1D/2D/3D and SphericalAdvection2D/3D: advect. Reconstruction1D/2D/3D, uv_center_flux and uv_node_flux: reconstruct. Diffusion2D/3D and SphericalDiffusion2D/3D: diffuse, diffusive_fluxes. BiharmonicDiffusion2D/3D and SphericalBiharmonicDiffusion2D/3D: biharmonic. MomentumAdvection2D/3D and SphericalMomentumAdvection2D/3D: advect_momentum(form=). gridax.analysis gradient_1d/2d/3d, divergence_1d/2d/3d, curl_2d/3d, laplacian_1d/2d/3d: gradient, divergence, curl, laplacian on a collocated grid. gridax.analysis adaptive_difference and the adaptive_gradient, adaptive_divergence, adaptive_curl and adaptive_laplacian functions: the same functions with mask=. bernoulli_potential, sw_potential_vorticity, qg_potential_vorticity and vertical_velocity take the grid instead of spacings.
* **boundary:** boundary presets named by physical role ([#544](https://github.com/jejjohnson/gridax/issues/544))
* **boundary:** regions of a field, interior, embed, select and apply_mask ([#543](https://github.com/jejjohnson/gridax/issues/543))
* **mask:** look up stencil tiers and coastal ghosts on the mask ([#537](https://github.com/jejjohnson/gridax/issues/537))
* **gridax:** divergence, curl, laplacian, gradient and grad_perp on every grid ([#533](https://github.com/jejjohnson/gridax/issues/533))
* **gridax:** canonical functions dispatch on the grid type, and the reductions are the first ([#529](https://github.com/jejjohnson/gridax/issues/529))
* **gridax:** masks are looked up by point, with wet(at=, kind=) and from_grid ([#527](https://github.com/jejjohnson/gridax/issues/527))
* **gridax:** fill_ghosts and BC take the spacing from the grid and the point from the field ([#524](https://github.com/jejjohnson/gridax/issues/524))
* **gridax:** the C-grid classes are CartesianCGrid* and SphericalCGrid* ([#521](https://github.com/jejjohnson/gridax/issues/521))
* **gridax:** grids name their axes and give spacing, coords and area at any point ([#520](https://github.com/jejjohnson/gridax/issues/520))
* **gridax:** `gridax_jax` and `finitevolx` no longer exist. Import the grid, mask and operator names from `gridax`, the solver names from `gridax.solvers` (`pip install "gridax[jax]"`) and the pipekit operators from `gridax.pipekit` (`pip install "gridax[pipekit]"`). The pre-#435 raw-stencil names and `gridax_jax.divergence_2d` / `relative_vorticity_cgrid` aliases are gone.
* **gridax:** the names above, and make_preconditioner, are gone from gridax_jax.

### Features

* **advection:** the vertical terms of a tracer, advect_vertical and diffuse_vertical ([#555](https://github.com/jejjohnson/gridax/issues/555)) ([bf59b84](https://github.com/jejjohnson/gridax/commit/bf59b843966f7c3417a855d52a343b77b4822bbd))
* **boundary:** boundary presets named by physical role ([#544](https://github.com/jejjohnson/gridax/issues/544)) ([e290006](https://github.com/jejjohnson/gridax/commit/e290006f5fdb4b320898b7527d9e3b4c1523bbd4))
* **boundary:** fill the dry points behind a coast ([#538](https://github.com/jejjohnson/gridax/issues/538)) ([ce7ca93](https://github.com/jejjohnson/gridax/commit/ce7ca93d77a106314b1897c8d42d40a41e02535f))
* **boundary:** regions of a field, interior, embed, select and apply_mask ([#543](https://github.com/jejjohnson/gridax/issues/543)) ([ce199b0](https://github.com/jejjohnson/gridax/commit/ce199b067e15afdad7d072a491db848ee599494f))
* **gridax:** a CField carries the point of the grid it sits on, and combine checks it ([#522](https://github.com/jejjohnson/gridax/issues/522)) ([7d4f473](https://github.com/jejjohnson/gridax/commit/7d4f4734738745d2650070250f46366b438d7b93))
* **gridax:** a CField crosses to xarray with xgcm's positions, and left faces shift to right ([#526](https://github.com/jejjohnson/gridax/issues/526)) ([c71d84f](https://github.com/jejjohnson/gridax/commit/c71d84f48a461887a70b497640ea36ddd77ec1d6))
* **gridax:** canonical functions dispatch on the grid type, and the reductions are the first ([#529](https://github.com/jejjohnson/gridax/issues/529)) ([a354072](https://github.com/jejjohnson/gridax/commit/a354072a6ae99d94c37660a0af9c49c934b1ba9e))
* **gridax:** collocated grids, with irregular axes, for fields at T ([#532](https://github.com/jejjohnson/gridax/issues/532)) ([1d546b1](https://github.com/jejjohnson/gridax/commit/1d546b19e0a879d1ce4fecd239c2f4d74b94a8e1))
* **gridax:** difference and interpolate take the move by its points, on every C-grid ([#530](https://github.com/jejjohnson/gridax/issues/530)) ([d68a2a6](https://github.com/jejjohnson/gridax/commit/d68a2a61d61fa77f6b31e3adcc5f8efe512ab260))
* **gridax:** divergence, curl, laplacian, gradient and grad_perp on every grid ([#533](https://github.com/jejjohnson/gridax/issues/533)) ([8eb4857](https://github.com/jejjohnson/gridax/commit/8eb4857cc203d87b87144152d4d135b95dd8debb))
* **gridax:** fill_ghosts and BC take the spacing from the grid and the point from the field ([#524](https://github.com/jejjohnson/gridax/issues/524)) ([32d7b38](https://github.com/jejjohnson/gridax/commit/32d7b38b78a7e77f5bb06f28d1a11055b5261904))
* **gridax:** fill_ghosts fills a state from a pytree of BCs ([#525](https://github.com/jejjohnson/gridax/issues/525)) ([22141ac](https://github.com/jejjohnson/gridax/commit/22141ac49dc13c2361743359e1361f487916d2c0))
* **gridax:** grids name their axes and give spacing, coords and area at any point ([#520](https://github.com/jejjohnson/gridax/issues/520)) ([5cc0b5d](https://github.com/jejjohnson/gridax/commit/5cc0b5daabb4979aaef8b8bfbca5af19fca65e95))
* **gridax:** masks are looked up by point, with wet(at=, kind=) and from_grid ([#527](https://github.com/jejjohnson/gridax/issues/527)) ([f3e2a57](https://github.com/jejjohnson/gridax/commit/f3e2a57cf10a0688bc25def92c2afed92d95b307))
* **gridax:** pipekit's dim functions work on a CField and keep its point ([#523](https://github.com/jejjohnson/gridax/issues/523)) ([81aef10](https://github.com/jejjohnson/gridax/commit/81aef10a46d815eb67abe60e846ea8d7c2f1a2bd))
* **gridax:** the canonical functions accept CField, infer its point and set the result's ([#531](https://github.com/jejjohnson/gridax/issues/531)) ([578dd24](https://github.com/jejjohnson/gridax/commit/578dd2462932ef6de6542d3056effce34f6933cb))
* **grid:** convert grids and fields between collocated and staggered ([#545](https://github.com/jejjohnson/gridax/issues/545)) ([a2f0ff4](https://github.com/jejjohnson/gridax/commit/a2f0ff4729904661fcff72955e6352288b8c4685))
* **mask:** look up stencil tiers and coastal ghosts on the mask ([#537](https://github.com/jejjohnson/gridax/issues/537)) ([76401ff](https://github.com/jejjohnson/gridax/commit/76401ffc77f6f2504bf705028898fcbfe2bfcc3d))
* **operators:** add GridOps, the canonical functions on one grid and mask ([#541](https://github.com/jejjohnson/gridax/issues/541)) ([ae0fc4f](https://github.com/jejjohnson/gridax/commit/ae0fc4f6635e80bd85743b8da5a29c8466fcf04a))
* **operators:** fill as many ghost layers as a scheme reads with bc= ([#539](https://github.com/jejjohnson/gridax/issues/539)) ([cd0f741](https://github.com/jejjohnson/gridax/commit/cd0f741f3354f048f1a203ffe3520b912adfa288))
* **operators:** make transport, rotation and diagnostics canonical functions ([#536](https://github.com/jejjohnson/gridax/issues/536)) ([494d73d](https://github.com/jejjohnson/gridax/commit/494d73d7b11b2607942d43b24dfb05f9a29f9437))
* **operators:** take scheme= on difference, interpolate and reconstruct ([#535](https://github.com/jejjohnson/gridax/issues/535)) ([b2bbab4](https://github.com/jejjohnson/gridax/commit/b2bbab458486ed24b70c99300e13d8aa1f3db9af))
* **operators:** vortex_force, with the scheme of Arakawa and Lamb (1981) ([#556](https://github.com/jejjohnson/gridax/issues/556)) ([9f7db99](https://github.com/jejjohnson/gridax/commit/9f7db999d7527794443ac0060959fb0663843639))
* **pipekit:** operators say how far they read, and graph_halo sums it ([#542](https://github.com/jejjohnson/gridax/issues/542)) ([4c114fd](https://github.com/jejjohnson/gridax/commit/4c114fd0daeb40ac0fcd6fe483d5357eb99aa35e))
* **schemes:** add Adaptive, with edge and coast policies ([#540](https://github.com/jejjohnson/gridax/issues/540)) ([e356a4f](https://github.com/jejjohnson/gridax/commit/e356a4f0f1de45d04222c001562afb3026da463f))
* **schemes:** add gridax.schemes with footprints, halos and exact weights ([#534](https://github.com/jejjohnson/gridax/issues/534)) ([4450596](https://github.com/jejjohnson/gridax/commit/445059630ce20fb661c0004ca7facc664efcf920))
* **schemes:** generate WENO of any odd order, and WENO-Z up to 9 ([#546](https://github.com/jejjohnson/gridax/issues/546)) ([fc2c629](https://github.com/jejjohnson/gridax/commit/fc2c629a7f97a781bcdeb15a3b4c2724ab478ddb))


### Bug Fixes

* **advection:** the face velocity wraps on periodic faces of advect ([#548](https://github.com/jejjohnson/gridax/issues/548)) ([83fcf59](https://github.com/jejjohnson/gridax/commit/83fcf59c68adb48830969e6495a05837c25bfca4))
* **operators:** no operation with a mask reads land ([#554](https://github.com/jejjohnson/gridax/issues/554)) ([d35143d](https://github.com/jejjohnson/gridax/commit/d35143d96e54885ef25ce5c36046523238e81f1d))
* **operators:** the collocated spherical laplacian is the flux form ([#549](https://github.com/jejjohnson/gridax/issues/549)) ([42fe52d](https://github.com/jejjohnson/gridax/commit/42fe52d9633f4bd08ec076cf014c4ee64fa5ca71))
* **pytree:** NumPy arrays join JAX arrays with array-api-compat installed ([#560](https://github.com/jejjohnson/gridax/issues/560)) ([33add80](https://github.com/jejjohnson/gridax/commit/33add80de944b71dd07b053570e24c5b411419ff))


### Code Refactoring

* **gridax:** every shim for a name that moved to another repo is removed ([#482](https://github.com/jejjohnson/gridax/issues/482)) ([ecddb62](https://github.com/jejjohnson/gridax/commit/ecddb623b3c3c2d1abdfac1a5b2507b1a1049bfc))
* **gridax:** gridax-jax folds into gridax behind the jax and pipekit extras ([#483](https://github.com/jejjohnson/gridax/issues/483)) ([5de277f](https://github.com/jejjohnson/gridax/commit/5de277fcdd42f3412262e4f37d1618f95b70bc44))
* **gridax:** the C-grid classes are CartesianCGrid* and SphericalCGrid* ([#521](https://github.com/jejjohnson/gridax/issues/521)) ([0defe5c](https://github.com/jejjohnson/gridax/commit/0defe5cf952e345772d8ed1525a4013979cf4cb0))
* **mask:** one builder for the masks of every rank, and reconstruct_fn goes ([#557](https://github.com/jejjohnson/gridax/issues/557)) ([557fc94](https://github.com/jejjohnson/gridax/commit/557fc9434dc74d2a5f3b8cdeb1bbf081169fbb83))
* remove the operator classes and the analysis twins, functions and GridOps remain ([#547](https://github.com/jejjohnson/gridax/issues/547)) ([e55e6da](https://github.com/jejjohnson/gridax/commit/e55e6dadf92f234dfe245c9893aadd43dce63c67)), closes [#493](https://github.com/jejjohnson/gridax/issues/493)

## [0.2.0](https://github.com/jejjohnson/gridax/compare/gridax-v0.1.0...gridax-v0.2.0) (2026-11-21)


### ⚠ BREAKING CHANGES

* **gridax:** spherical results move by about 3e-5 relative with the new default radius; pass R=6_371_200.0 to keep the old numbers.

### Features

* **gridax:** fourth- and sixth-order staggered differences and averages, staggered_difference and staggered_average ([#469](https://github.com/jejjohnson/gridax/issues/469)) ([bf5ac89](https://github.com/jejjohnson/gridax/commit/bf5ac8991a8350e487867b884b7c23a76c7d0f83)), closes [#457](https://github.com/jejjohnson/gridax/issues/457)
* **gridax:** gridax.stencils, one stencil module for collocated and staggered points, with exact coefficients ([#465](https://github.com/jejjohnson/gridax/issues/465)) ([87dce56](https://github.com/jejjohnson/gridax/commit/87dce5655fbb43b48a6745c3dd36e40322168792)), closes [#454](https://github.com/jejjohnson/gridax/issues/454)
* **gridax:** the stencil module solves for cell averages and WENO, with data="cell", biased offsets, optimal weights and smoothness matrices ([#468](https://github.com/jejjohnson/gridax/issues/468)) ([76bc36f](https://github.com/jejjohnson/gridax/commit/76bc36f068b6f67834623b3bce6682fc3a7a1fc8)), closes [#462](https://github.com/jejjohnson/gridax/issues/462)


### Code Refactoring

* **gridax:** the constants, the Coriolis helpers and the linear equation of state come from geophysix ([#459](https://github.com/jejjohnson/gridax/issues/459)) ([4d2253d](https://github.com/jejjohnson/gridax/commit/4d2253de3fbe32fc93ec272f6344a17312ba603e)), closes [#446](https://github.com/jejjohnson/gridax/issues/446)

## 0.1.0 (2026-11-20)


### Features

* first release: grids, masks, the raw stencils, the C-grid operators, advection, diffusion, boundary conditions, reductions and gridax.analysis, on NumPy, JAX or any array-API backend ([#435](https://github.com/jejjohnson/finitevolX/issues/435))
* **gridax:** analysis kernels on collocated grids — finite differences and axis remapping on any array backend ([#434](https://github.com/jejjohnson/finitevolX/issues/434)) ([95798a1](https://github.com/jejjohnson/finitevolX/commit/95798a1a7d7d4052bd4d6bea634f1e0322f39f64))


### Bug Fixes

* **grid:** spherical T rows are cell centres, so whole-sphere area weights sum to 4πR² ([#433](https://github.com/jejjohnson/finitevolX/issues/433)) ([067d292](https://github.com/jejjohnson/finitevolX/commit/067d292cac70ff7eade615aaadc7d86be1039b5b)), closes [#322](https://github.com/jejjohnson/finitevolX/issues/322)


### Code Refactoring

* rename gridax-ops to gridax-jax (import gridax_jax) ([#437](https://github.com/jejjohnson/finitevolX/issues/437)) ([aff7146](https://github.com/jejjohnson/finitevolX/commit/aff714687ebebc14fd1ad06a8d11f520124a6146)), closes [#435](https://github.com/jejjohnson/finitevolX/issues/435)
