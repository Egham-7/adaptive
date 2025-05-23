## [1.4.1](https://github.com/grafana/profiles-drilldown/compare/v1.4.0...v1.4.1) (2025-05-21)


### Features

* **GitHub Integration:** Add a github callback view ([#495](https://github.com/grafana/profiles-drilldown/issues/495)) ([012a0e6](https://github.com/grafana/profiles-drilldown/commit/012a0e6a775fd3af777ccae5c538e73bd83d4833))



# [1.4.0](https://github.com/grafana/profiles-drilldown/compare/v1.3.0...v1.4.0) (2025-05-15)


### Features

* Add extension point to add additional settings ([#478](https://github.com/grafana/profiles-drilldown/issues/478)) ([4ff758f](https://github.com/grafana/profiles-drilldown/commit/4ff758f4b2d0a458da647b663b2488140a0c0b63))



# [1.3.0](https://github.com/grafana/explore-profiles/compare/v1.2.3...v1.3.0) (2025-04-16)


### Bug Fixes

* Cascader options break when services have multiple levels with matching prefix ([#484](https://github.com/grafana/explore-profiles/issues/484)) ([2bb6034](https://github.com/grafana/explore-profiles/commit/2bb603464a329bec9b6da2c26ca0c30a2ef6f45e))
* **Settings:** Reload settings when data source changes ([#474](https://github.com/grafana/explore-profiles/issues/474)) ([23d814b](https://github.com/grafana/explore-profiles/commit/23d814be30fc8cf050024e556c132785d24a196c))


### Features

* **CI:** Append SHA to version number when releasing from main ([#455](https://github.com/grafana/explore-profiles/issues/455)) ([03e6158](https://github.com/grafana/explore-profiles/commit/03e6158377a87b6494ac39a9764296dd2de19a01))
* **Metrics:** Basic UI for experimental metrics from profiles ([#462](https://github.com/grafana/explore-profiles/issues/462)) ([1500a59](https://github.com/grafana/explore-profiles/commit/1500a59e6daeb097576449273e2dc3da47f8f311))



## [1.2.3](https://github.com/grafana/explore-profiles/compare/v1.2.2...v1.2.3) (2025-04-03)


### Bug Fixes

* Get suggestions for optimize code ([#480](https://github.com/grafana/explore-profiles/issues/480)) ([23674c3](https://github.com/grafana/explore-profiles/commit/23674c3eaf8adf5572a725bade08f0a53038e77b))



## [1.2.2](https://github.com/grafana/explore-profiles/compare/v1.2.1...v1.2.2) (2025-04-03)


### Bug Fixes

* revert "profiling insights" extension ([#447](https://github.com/grafana/explore-profiles/issues/447)) ([43b35dc](https://github.com/grafana/explore-profiles/commit/43b35dc8f929e922bb8d554b92c6a763581d3443))



## [1.2.1](https://github.com/grafana/explore-profiles/compare/v1.2.0...v1.2.1) (2025-04-02)


### Bug Fixes

* **FlameGraph:** Use entire file as context in "Optimize Code" prompt ([#459](https://github.com/grafana/explore-profiles/issues/459)) ([569c17c](https://github.com/grafana/explore-profiles/commit/569c17c41511167add7a7a4db3f90cbc88897d68))
* **llm:** Add extra context in AI prompt about other lines in a file ([#460](https://github.com/grafana/explore-profiles/issues/460)) ([004b7aa](https://github.com/grafana/explore-profiles/commit/004b7aa04a136b02f036c0e33a27d1b333cbbf93))


### Features

* "profiling insights" extension ([#447](https://github.com/grafana/explore-profiles/issues/447)) ([13ff552](https://github.com/grafana/explore-profiles/commit/13ff552083cc4555aa21e8401be1516a364b8271))



# [1.2.0](https://github.com/grafana/explore-profiles/compare/v1.1.0...v1.2.0) (2025-03-10)


### Features

* Add generated typed Pyroscope API client ([#410](https://github.com/grafana/explore-profiles/issues/410)) ([2e2b4e1](https://github.com/grafana/explore-profiles/commit/2e2b4e1d41fab1ed37029fff0073879f51fe932f))
* **FlameGraph:** Show the flame graph for a selected span ([#445](https://github.com/grafana/explore-profiles/issues/445)) ([2e036ee](https://github.com/grafana/explore-profiles/commit/2e036eeabc705e147560f13d8137821adf2585c5))



# [1.1.0](https://github.com/grafana/explore-profiles/compare/v1.0.0...v1.1.0) (2025-02-21)


### Bug Fixes

* **Routes:** Add redirection for non-supported paths ([#427](https://github.com/grafana/explore-profiles/issues/427)) ([edd56d3](https://github.com/grafana/explore-profiles/commit/edd56d34cb40dfc8bfbaba6788c6c04f46d571b2))


### Features

* **investigations:** adapt to changed plugin id ([#429](https://github.com/grafana/explore-profiles/issues/429)) ([7b41af9](https://github.com/grafana/explore-profiles/commit/7b41af95c9a938d05c008017b5546a5aab319531))
* Rename Explore Profiles to Grafana Profiles Drilldown ([#430](https://github.com/grafana/explore-profiles/issues/430)) ([e91ff90](https://github.com/grafana/explore-profiles/commit/e91ff90cf697070be37b2140d3318ea4135db7f4)), closes [#432](https://github.com/grafana/explore-profiles/issues/432)



# [1.0.0](https://github.com/grafana/explore-profiles/compare/v0.1.23...v1.0.0) (2025-02-14)


### Features

* **App:** Changes for GA ([#398](https://github.com/grafana/explore-profiles/issues/398)) ([71ed3f8](https://github.com/grafana/explore-profiles/commit/71ed3f8d312317474ffec674d056968f3fefad64))



## [0.1.23](https://github.com/grafana/explore-profiles/compare/v0.1.22...v0.1.23) (2025-02-14)


### Bug Fixes

* Remove lazy loading of page components ([#422](https://github.com/grafana/explore-profiles/issues/422)) ([45bc71e](https://github.com/grafana/explore-profiles/commit/45bc71e27ac8306d735c980b3436b61889f522d9))



## [0.1.22](https://github.com/grafana/explore-profiles/compare/v0.1.21...v0.1.22) (2025-02-13)


### Bug Fixes

* **FlameGraph:** Always render an error message when loading fails ([#407](https://github.com/grafana/explore-profiles/issues/407)) ([c531bf2](https://github.com/grafana/explore-profiles/commit/c531bf2296ac75da8938ce6af1ec1660dd6e7b73))


### Features

* **Tracking:** Add page view tracking ([#408](https://github.com/grafana/explore-profiles/issues/408)) ([cf5b786](https://github.com/grafana/explore-profiles/commit/cf5b786ada6e211af2385192b9f8059906f935f5))



## [0.1.21](https://github.com/grafana/explore-profiles/compare/v0.1.20...v0.1.21) (2025-02-12)


### Bug Fixes

* **Faro:** Narrow down frontend logging to plugin URL ([#395](https://github.com/grafana/explore-profiles/issues/395)) ([8d011ce](https://github.com/grafana/explore-profiles/commit/8d011cedab660fe23b0711b303ef3fe450b78614))



## [0.1.20](https://github.com/grafana/explore-profiles/compare/v0.1.19...v0.1.20) (2025-02-06)


### Bug Fixes

* **Filters:** Ensure filters are always cleared when changing the data source or the service ([#374](https://github.com/grafana/explore-profiles/issues/374)) ([1c7b6c3](https://github.com/grafana/explore-profiles/commit/1c7b6c3f537b9ff0bc9060bac5f0e2c240ba00c8))
* **FunctionDetailsPanel:** Fix start ellipsis for file names containing non-alpha chars ([#373](https://github.com/grafana/explore-profiles/issues/373)) ([f79a1b7](https://github.com/grafana/explore-profiles/commit/f79a1b7628d8dd0bfd478122a0296dd2a1c72948))
* **GitHubIntegration:** Handle function details for inlining ([#347](https://github.com/grafana/explore-profiles/issues/347)) ([5e28b3c](https://github.com/grafana/explore-profiles/commit/5e28b3c0e571f254915d08e0a166a66d33ff240b))
* **Grid:** Fix error message display ([#359](https://github.com/grafana/explore-profiles/issues/359)) ([1ca4ff2](https://github.com/grafana/explore-profiles/commit/1ca4ff2795cfa8118ec9e0d12dcad4b04d336534))
* **Settings:** Prevent warning to be displayed when no settings is returned by the Settings API ([#384](https://github.com/grafana/explore-profiles/issues/384)) ([cdce58a](https://github.com/grafana/explore-profiles/commit/cdce58a48bad24be1cf64ed81658bbcd0afca521))


### Features

* **AdHocView:** Remove Grafana menu item to the Ad Hoc view ([#385](https://github.com/grafana/explore-profiles/issues/385)) ([5c265fb](https://github.com/grafana/explore-profiles/commit/5c265fb4b5273083b6150aad46d32a6f44027fda))
* **LabelsView:** Add maxima visualizations ([#361](https://github.com/grafana/explore-profiles/issues/361)) ([25095c6](https://github.com/grafana/explore-profiles/commit/25095c6882d17ba6d88e7f5f861d5560e112a930))
* **LabelsView:** update main time series when a "group by" label is selected ([#341](https://github.com/grafana/explore-profiles/issues/341)) ([775b37d](https://github.com/grafana/explore-profiles/commit/775b37dc6ef9433b3f5674f56869a37a16892871))


### Performance Improvements

* **Series:** Limit the number of series request by breakdown charts ([#219](https://github.com/grafana/explore-profiles/issues/219)) ([8e1161d](https://github.com/grafana/explore-profiles/commit/8e1161df84283eecadc8aa9da55e46eb401a1e8c))



## [0.1.19](https://github.com/grafana/explore-profiles/compare/v0.1.18...v0.1.19) (2025-01-27)


### Bug Fixes

* **DiffFlameGraph:** Always disable time ranges sync before applying a preset ([#355](https://github.com/grafana/explore-profiles/issues/355)) ([272a98b](https://github.com/grafana/explore-profiles/commit/272a98bc13ffe7e93673c41ab1795b12ae2fbb3e))



## [0.1.18](https://github.com/grafana/explore-profiles/compare/v0.1.17...v0.1.18) (2025-01-23)


### Bug Fixes

* **DiffFlameGraphView:** Clear preset option when applying auto-select ([#313](https://github.com/grafana/explore-profiles/issues/313)) ([b0f4001](https://github.com/grafana/explore-profiles/commit/b0f400163378136b1d1a8485877e1fc596b1d222))
* **DiffFlameGraphView:** Preserve context after leaving/re-entering the view ([#319](https://github.com/grafana/explore-profiles/issues/319)) ([367ddab](https://github.com/grafana/explore-profiles/commit/367ddabe175c6a942b3ec2371d01a3013de1cccc))
* **Header:** Prevent crash if useChromeHeaderHeight is not available (for Grafana < 11.3) ([#312](https://github.com/grafana/explore-profiles/issues/312)) ([c638416](https://github.com/grafana/explore-profiles/commit/c6384162e1c3dc5f068dd814ad5431e23efb8537))
* **QueryRunners:** Prevent invalid queries to run ([#316](https://github.com/grafana/explore-profiles/issues/316)) ([5f5046c](https://github.com/grafana/explore-profiles/commit/5f5046cb82705c7ddeb2e80934becf3f8b887918))


### Features

* Add investigations support ([#301](https://github.com/grafana/explore-profiles/issues/301)) ([7f95852](https://github.com/grafana/explore-profiles/commit/7f958526d6af6d494a3f1095a89404d4fcacca10))
* Add query link extension ([#220](https://github.com/grafana/explore-profiles/issues/220)) ([62720ad](https://github.com/grafana/explore-profiles/commit/62720ad1cbd1346c772a521c133988d043851f55))
* **DiffView:** Time ranges sync ([#288](https://github.com/grafana/explore-profiles/issues/288)) ([45cea14](https://github.com/grafana/explore-profiles/commit/45cea14eef9fbde49f4474051a5833a1be623d2a))
* **FlameGraph:** Keep items focused when data changes ([#336](https://github.com/grafana/explore-profiles/issues/336)) ([d8ff887](https://github.com/grafana/explore-profiles/commit/d8ff887d0b9207c02a3a7563e845dab7df0b94d5))
* **GitHubIntegration:** Add info tooltip on connect button ([#328](https://github.com/grafana/explore-profiles/issues/328)) ([a929ddd](https://github.com/grafana/explore-profiles/commit/a929ddd56ea983ad5b5210c3e276afc778d7cdba))
* **Timeseries:** Add open in Explore menu item ([#300](https://github.com/grafana/explore-profiles/issues/300)) ([a9b0891](https://github.com/grafana/explore-profiles/commit/a9b0891aa2bc3666b57b2b9ab06e1b704dde8e85))
* Upgrade Grafana to v11.3.0 ([#287](https://github.com/grafana/explore-profiles/issues/287)) ([595a1cc](https://github.com/grafana/explore-profiles/commit/595a1cca4a83f84341c258f823a2cd2b61659268))


### Performance Improvements

* Lazy load page components ([#324](https://github.com/grafana/explore-profiles/issues/324)) ([c0ffd33](https://github.com/grafana/explore-profiles/commit/c0ffd33b3f1c2b6d9ae960429ebf23d1436bdde3))


### Reverts

* Revert "refactor(*): Lazy load page components (#322)" (#323) ([362e02b](https://github.com/grafana/explore-profiles/commit/362e02b6a8356dd9366259a56d4d1c56b3b522a0)), closes [#322](https://github.com/grafana/explore-profiles/issues/322) [#323](https://github.com/grafana/explore-profiles/issues/323)



## [0.1.17](https://github.com/grafana/explore-profiles/compare/v0.1.16...v0.1.17) (2024-11-19)


### Bug Fixes

* **ServiceDropdown:** Retrieve last used service name only if it's not provided in the URL ([#284](https://github.com/grafana/explore-profiles/issues/284)) ([28ca16e](https://github.com/grafana/explore-profiles/commit/28ca16ea5cd1340019278756fd9cfa7f583df268))



## [0.1.16](https://github.com/grafana/explore-profiles/compare/v0.1.15...v0.1.16) (2024-11-14)


### Bug Fixes

* **QueryBuilder:** Prevent invalid filters to be used after parsing ([#276](https://github.com/grafana/explore-profiles/issues/276)) ([e6cac6e](https://github.com/grafana/explore-profiles/commit/e6cac6ed097a1ce0609d4fc83400de2303f33fec))
* **Tracking:** Ensure select action type is tracked ([#278](https://github.com/grafana/explore-profiles/issues/278)) ([9527644](https://github.com/grafana/explore-profiles/commit/952764487d676408ce05b741902ada9ad3571b8b))
* **Tracking:** Use custom reporter ([#277](https://github.com/grafana/explore-profiles/issues/277)) ([dfbb3a6](https://github.com/grafana/explore-profiles/commit/dfbb3a6a6c8f177d9ecf66a52fbb2b7794ea686f))


### Features

* **Export:** Clarify that export to flamegraph.com option will create a public URL ([#275](https://github.com/grafana/explore-profiles/issues/275)) ([c5a0962](https://github.com/grafana/explore-profiles/commit/c5a0962bdf4cd19d9427515384e9c106b9dedd07))
* **Export:** Disable export to flame graph.com ([#280](https://github.com/grafana/explore-profiles/issues/280)) ([e631055](https://github.com/grafana/explore-profiles/commit/e63105522393c096feceb3f0a66cd9e1e0934a56))



## [0.1.15](https://github.com/grafana/explore-profiles/compare/v0.1.14...v0.1.15) (2024-11-05)


### Bug Fixes

* **DiffView:** Clicking on "Auto-select" selects a 25% range ([#254](https://github.com/grafana/explore-profiles/issues/254)) ([9b3dd8a](https://github.com/grafana/explore-profiles/commit/9b3dd8a16a961d319fe4e7190f772527d0d9e5e0))
* **DiffView:** Disable AI button when no selections ([#258](https://github.com/grafana/explore-profiles/issues/258)) ([58e89cb](https://github.com/grafana/explore-profiles/commit/58e89cb20d7bbfe08e5cfcabc00bb8dd6030f581))
* **DiffView:** Ensure ranges are initialized when landing ([#233](https://github.com/grafana/explore-profiles/issues/233)) ([4f95549](https://github.com/grafana/explore-profiles/commit/4f955491f9bf795aedb9d6aa79e7ba08d5379e7b))
* **DiffView:** Fix headers wrap ([#259](https://github.com/grafana/explore-profiles/issues/259)) ([6ada58a](https://github.com/grafana/explore-profiles/commit/6ada58afe7f981ed71ec68e9481e29a95633e13a))
* **DiffView:** Fix incorrect preset label ([#257](https://github.com/grafana/explore-profiles/issues/257)) ([a738694](https://github.com/grafana/explore-profiles/commit/a738694e5f6618f0efce7654852fb1189a2c920d))
* **ExplorationSelector:** Fix background color ([#255](https://github.com/grafana/explore-profiles/issues/255)) ([46db256](https://github.com/grafana/explore-profiles/commit/46db25677c0f54150f55a8fdfaa051d9ace33ab3))
* **FunctionDetails:** Correctly render blank lines ([ec3ed5c](https://github.com/grafana/explore-profiles/commit/ec3ed5c915d317c9ae6c15c4f1c00dc12cfaac39))
* **GitHubIntegration:** Fix "Learn more" href ([#245](https://github.com/grafana/explore-profiles/issues/245)) ([1848159](https://github.com/grafana/explore-profiles/commit/184815953a3fb5c20b6586df59b89b5b948695b6))
* **LabelValuesGrid:** decrease column size to accommodate small screen resolutions ([#235](https://github.com/grafana/explore-profiles/issues/235)) ([f62b17a](https://github.com/grafana/explore-profiles/commit/f62b17a32f91fe97cc133f93204ae365837fb1e3))
* **OnboardingModal:** Change Grafana Agent to Grafana Alloy ([#256](https://github.com/grafana/explore-profiles/issues/256)) ([27453ed](https://github.com/grafana/explore-profiles/commit/27453edef4051ea2ae4399b9502f1ed2aa18081c))
* **SettingsView:** Fix back button after modifying the max nodes setting ([#234](https://github.com/grafana/explore-profiles/issues/234)) ([673b44c](https://github.com/grafana/explore-profiles/commit/673b44cffe5e00378780da40d4d8c94d8289e171))
* **ShareableUrl:** Fix when the default time range is selected ([#244](https://github.com/grafana/explore-profiles/issues/244)) ([ae6ddeb](https://github.com/grafana/explore-profiles/commit/ae6ddeb842066fb8513a1ebd4c50133bf8a41b69))
* Small UI fixes ([#248](https://github.com/grafana/explore-profiles/issues/248)) ([ee881fa](https://github.com/grafana/explore-profiles/commit/ee881fa06929fa317299a0e24b6eca4e228cb571))
* **Timeseries:** Persist scale when data changes ([#251](https://github.com/grafana/explore-profiles/issues/251)) ([8cb6ced](https://github.com/grafana/explore-profiles/commit/8cb6ced4272a826f2a63e921eda2f62ee4d1a18d))


### Features

* **AppHeader:** Revamp header ([#230](https://github.com/grafana/explore-profiles/issues/230)) ([f482d7b](https://github.com/grafana/explore-profiles/commit/f482d7b2e4c77c7e88061e7198a85e5fd778c47b))
* **DiffFlameGraph:** Add "how to" infos ([#228](https://github.com/grafana/explore-profiles/issues/228)) ([494b659](https://github.com/grafana/explore-profiles/commit/494b659eb983231b9971429009b9185c7e5203a3))
* **DiffView:** Add CTAs and comparison presets ([#231](https://github.com/grafana/explore-profiles/issues/231)) ([e8bbf2e](https://github.com/grafana/explore-profiles/commit/e8bbf2e9e40dfc6e0eec6e4e8aa9ec35917d04d1))
* **LabelsView:** Include/exclude panel actions ([#210](https://github.com/grafana/explore-profiles/issues/210)) ([2c2d5f5](https://github.com/grafana/explore-profiles/commit/2c2d5f59b54ae2e49cd67dc5c5ea265b21e9a53f))
* **TimeSeries:** Add menu with scale options ([#249](https://github.com/grafana/explore-profiles/issues/249)) ([06b71d1](https://github.com/grafana/explore-profiles/commit/06b71d16190f9fd081c6f980ca580f58b1a1d2c5))



## [0.1.14](https://github.com/grafana/explore-profiles/compare/v0.1.13...v0.1.14) (2024-10-17)


### Bug Fixes

* **Faro:** Filter out events not related to the app ([#225](https://github.com/grafana/explore-profiles/issues/225)) ([57a7c58](https://github.com/grafana/explore-profiles/commit/57a7c581988d8418baa58ec872780e98f4703733))
* **Header:** Fix sticky header position in Grafana v11.3+ ([#218](https://github.com/grafana/explore-profiles/issues/218)) ([a4f226f](https://github.com/grafana/explore-profiles/commit/a4f226ff44c8b991efe1ab64cf4a4557f2dba903))


### Features

* **ServiceNameVariable:** Persist last service selected in localStorage ([#222](https://github.com/grafana/explore-profiles/issues/222)) ([3917660](https://github.com/grafana/explore-profiles/commit/39176600a4c260aa33499635954df5174fc6a54a))



## [0.1.13](https://github.com/grafana/explore-profiles/compare/v0.1.12...v0.1.13) (2024-10-08)


### Features

* **UI:** Another batch of minor improvements ([#213](https://github.com/grafana/explore-profiles/issues/213)) ([8419560](https://github.com/grafana/explore-profiles/commit/8419560160ce99824eaa3f56889a50a9a01b5919))



## [0.1.12](https://github.com/grafana/explore-profiles/compare/v0.1.11...v0.1.12) (2024-10-04)


### Bug Fixes

* **Code:** do not show Optimize Code button when no code is available ([#208](https://github.com/grafana/explore-profiles/issues/208)) ([6af234d](https://github.com/grafana/explore-profiles/commit/6af234d6a4fec65a1c51adf9439cd28802462173))
* **Filters:** ensure "is empty" filter is synced with URL ([#205](https://github.com/grafana/explore-profiles/issues/205)) ([8fc8fc4](https://github.com/grafana/explore-profiles/commit/8fc8fc4fdc31860cd1d1def3ca7f603ffe5b10fe))
* **QueryBuilder:** Filters with regex values can be edited in place ([#207](https://github.com/grafana/explore-profiles/issues/207)) ([75de5e2](https://github.com/grafana/explore-profiles/commit/75de5e291cb46930e18e7f41fef6166ec69fa341))


### Features

* Minor improvements ([#211](https://github.com/grafana/explore-profiles/issues/211)) ([0486f33](https://github.com/grafana/explore-profiles/commit/0486f338404915525f53ae26b9723ac0455e216a))
* **QueryBuilder:** Enable "in"/"not in" operators ([#122](https://github.com/grafana/explore-profiles/issues/122)) ([9574828](https://github.com/grafana/explore-profiles/commit/9574828fead1168e0d143de49357f5997c5eaf5f))
* **StatsPanel:** Add title on hover value + vertical border to separate compare actions ([#212](https://github.com/grafana/explore-profiles/issues/212)) ([71a29e5](https://github.com/grafana/explore-profiles/commit/71a29e506596d3f9207cab2ae9f658ff079c088e))



## [0.1.11](https://github.com/grafana/explore-profiles/compare/v0.1.10...v0.1.11) (2024-09-30)


### Features

* Minor UI improvements (timeseries point size, plugin info tooltip) ([#194](https://github.com/grafana/explore-profiles/issues/194)) ([621982a](https://github.com/grafana/explore-profiles/commit/621982a990c320594c6764e1ce035aacde652ac8))
* **QuickFilter:** Add results count ([#193](https://github.com/grafana/explore-profiles/issues/193)) ([dc4012d](https://github.com/grafana/explore-profiles/commit/dc4012d39aed84561fb7d14726e426aad95544b6))



## [0.1.10](https://github.com/grafana/explore-profiles/compare/v0.1.9...v0.1.10) (2024-09-25)



## [0.1.9](https://github.com/grafana/explore-profiles/compare/v0.1.8...v0.1.9) (2024-09-17)


### Bug Fixes

* **DiffFlameGraph:** Remove non-working pprof export ([#169](https://github.com/grafana/explore-profiles/issues/169)) ([662cd48](https://github.com/grafana/explore-profiles/commit/662cd488ae7a41e4843ca66694743d4777ac1230))
* **ExplainFlameGraph:** Add tooltip when the LLM plugin is not installed ([#163](https://github.com/grafana/explore-profiles/issues/163)) ([d395391](https://github.com/grafana/explore-profiles/commit/d3953913be2b072ddac8413fd9341a43dc4f865e))
* **Faro:** Fix Faro SDK config ([#174](https://github.com/grafana/explore-profiles/issues/174)) ([3ed6362](https://github.com/grafana/explore-profiles/commit/3ed636207927c0423cbca8c40444ba57cd6885b2))
* Fix useUrlSearchParams ([#171](https://github.com/grafana/explore-profiles/issues/171)) ([179b060](https://github.com/grafana/explore-profiles/commit/179b0608d5d99fa5370dcee52c8431259de1da1f))
* **LabelsDataSource:** Limit the maximum number of concurrent requests to fetch label values ([#165](https://github.com/grafana/explore-profiles/issues/165)) ([cb8149c](https://github.com/grafana/explore-profiles/commit/cb8149c36f4f40502b62d1f4ed96c46ea10a2c65))


### Features

* Add give feeback button and preview badge ([#167](https://github.com/grafana/explore-profiles/issues/167)) ([a23fa61](https://github.com/grafana/explore-profiles/commit/a23fa61b8982b0c8877a0b70ecab26747d1e4fa0))
* **AppHeader:** Add Settings button ([#172](https://github.com/grafana/explore-profiles/issues/172)) ([9d7fb6b](https://github.com/grafana/explore-profiles/commit/9d7fb6b08b9bea0cfebe6a74c883d8ff92cc9ad9))
* Remove legacy comparison views code ([#143](https://github.com/grafana/explore-profiles/issues/143)) ([816363f](https://github.com/grafana/explore-profiles/commit/816363faea2dcbf10789bb68a50b3e85947fc2a4))
* Upgrade Grafana to v11.2.0 ([#173](https://github.com/grafana/explore-profiles/issues/173)) ([15680e6](https://github.com/grafana/explore-profiles/commit/15680e6b810f7771e9a874b0cacd6d093403354d))



## [0.1.8](https://github.com/grafana/explore-profiles/compare/v0.1.6...v0.1.8) (2024-09-11)


### Bug Fixes

* **Labels:** Fix error with bar gauges viz and new Grafana version ([#159](https://github.com/grafana/explore-profiles/issues/159)) ([b527961](https://github.com/grafana/explore-profiles/commit/b52796103af9db785d681fdac22bf6d751a7f734))


### Features

* Add histogram visualizations ([#141](https://github.com/grafana/explore-profiles/issues/141)) ([2265be7](https://github.com/grafana/explore-profiles/commit/2265be70ea67cfdc44aad33e1a1f7951076db815))
* create new browser history entry on some user actions  ([#128](https://github.com/grafana/explore-profiles/issues/128)) ([5439ab3](https://github.com/grafana/explore-profiles/commit/5439ab32f0e4a21f3affbe6bfbe12da7cacd12b1))
* **DiffFlameGraph:** Add flame graph range in timeseries legend ([#140](https://github.com/grafana/explore-profiles/issues/140)) ([8729c31](https://github.com/grafana/explore-profiles/commit/8729c31dddf383d2d6ca4c2178397045c31d9654))
* **GitHubIntegration:** Migrate GitHub integration to Scenes ([#142](https://github.com/grafana/explore-profiles/issues/142)) ([0386bbc](https://github.com/grafana/explore-profiles/commit/0386bbc369538763c69fce1cc07a45fb82619beb))
* support submodules for GitHub Integration ([#147](https://github.com/grafana/explore-profiles/issues/147)) ([52ecea8](https://github.com/grafana/explore-profiles/commit/52ecea89b5a436b3dc03ff352127f55ea315e037))



## [0.1.7](https://github.com/grafana/explore-profiles/compare/v0.1.6...v0.1.7) (2024-08-29)


### Features

* Add histogram visualizations ([#141](https://github.com/grafana/explore-profiles/issues/141)) ([2265be7](https://github.com/grafana/explore-profiles/commit/2265be70ea67cfdc44aad33e1a1f7951076db815))
* create new browser history entry on some user actions  ([#128](https://github.com/grafana/explore-profiles/issues/128)) ([5439ab3](https://github.com/grafana/explore-profiles/commit/5439ab32f0e4a21f3affbe6bfbe12da7cacd12b1))
* **DiffFlameGraph:** Add flame graph range in timeseries legend ([#140](https://github.com/grafana/explore-profiles/issues/140)) ([8729c31](https://github.com/grafana/explore-profiles/commit/8729c31dddf383d2d6ca4c2178397045c31d9654))
* **GitHubIntegration:** Migrate GitHub integration to Scenes ([#142](https://github.com/grafana/explore-profiles/issues/142)) ([0386bbc](https://github.com/grafana/explore-profiles/commit/0386bbc369538763c69fce1cc07a45fb82619beb))



## [0.1.6](https://github.com/grafana/explore-profiles/compare/v0.1.5...v0.1.6) (2024-08-27)


### Bug Fixes

* **Ci:** Fix docker compose commands ([#111](https://github.com/grafana/explore-profiles/issues/111)) ([4ee541a](https://github.com/grafana/explore-profiles/commit/4ee541acbe822d92abfc9344eda4611600b1476e))
* **DiffFlameGraph:** Fix the "Explain Flame Graph" (AI) feature ([#129](https://github.com/grafana/explore-profiles/issues/129)) ([a40c02b](https://github.com/grafana/explore-profiles/commit/a40c02b7c37ac309d878689c5929ef770900d6f5))
* **Favorites:** Render "No results" when there are no favorites ([#101](https://github.com/grafana/explore-profiles/issues/101)) ([426469d](https://github.com/grafana/explore-profiles/commit/426469d239b9ac86ad7e6fe4a21385836926a264))
* **Labels:** Fix "Discarded by user" error in the UI ([#110](https://github.com/grafana/explore-profiles/issues/110)) ([2e9baab](https://github.com/grafana/explore-profiles/commit/2e9baab391168022f4de7bf3933e8ba4baac95df))
* **SceneLabelValuePanel:** Fix border color when baseline/comparison is selected ([#123](https://github.com/grafana/explore-profiles/issues/123)) ([5b4058a](https://github.com/grafana/explore-profiles/commit/5b4058a90ac6f713d50c9686813f273233dc4a39))
* **ScenesProfileExplorer:** Make labels more responsive on smaller screens ([10c97dc](https://github.com/grafana/explore-profiles/commit/10c97dc69714a6a0f97bbaa086dd7263e8e72950))


### Features

* **CompareView:** Implement new Comparison view with Scenes ([#119](https://github.com/grafana/explore-profiles/issues/119)) ([127d6c3](https://github.com/grafana/explore-profiles/commit/127d6c3f952d1e679bcb29c6e2d62ca9d1eed51f))
* **FlameGraph:** Add missing export menu ([#132](https://github.com/grafana/explore-profiles/issues/132)) ([f57b0ca](https://github.com/grafana/explore-profiles/commit/f57b0ca5329b0b2a7e58f7387391299475ddc952))
* **Labels:** Improve comparison flow ([#117](https://github.com/grafana/explore-profiles/issues/117)) ([31d0632](https://github.com/grafana/explore-profiles/commit/31d06326fa9e82a906635ac371a9e206cfa2bb54))
* **Timeseries:** Add total resource consumption in legend ([#108](https://github.com/grafana/explore-profiles/issues/108)) ([1fbb2df](https://github.com/grafana/explore-profiles/commit/1fbb2dfbc1d0a5d837afa74c4783171aded0258a))



## [0.1.5](https://github.com/grafana/explore-profiles/compare/v0.1.4...v0.1.5) (2024-07-29)


### Features

* **Labels:** Various UI/UX improvements ([#93](https://github.com/grafana/explore-profiles/issues/93)) ([bddad3c](https://github.com/grafana/explore-profiles/commit/bddad3cf21e6e1459eed03167c8c6f6d24e802d4))
* Revamp exploration type selector ([#94](https://github.com/grafana/explore-profiles/issues/94)) ([60dab67](https://github.com/grafana/explore-profiles/commit/60dab67af27f7ec72a3e9de11885f901937c23ed))



## [0.1.4](https://github.com/grafana/explore-profiles/compare/v0.1.3...v0.1.4) (2024-07-25)


### Bug Fixes

* **Onboarding:** Handle gracefully when there's no data source configured ([#76](https://github.com/grafana/explore-profiles/issues/76)) ([4c18444](https://github.com/grafana/explore-profiles/commit/4c1844498d8b3bde4bb5b30ac889419b7462fb8b))
* **PanelTitle:** Remove series count when only 1 serie ([#78](https://github.com/grafana/explore-profiles/issues/78)) ([8422e6d](https://github.com/grafana/explore-profiles/commit/8422e6d2b2d8e21d0178ed20599ce13e16194da5))
* **SceneByVariableRepeaterGrid:** Prevent extra renders ([#86](https://github.com/grafana/explore-profiles/issues/86)) ([bf14755](https://github.com/grafana/explore-profiles/commit/bf1475580f68beec434287283d079d0fed250cad))


### Features

* Avoid no data panels ([#80](https://github.com/grafana/explore-profiles/issues/80)) ([72120b7](https://github.com/grafana/explore-profiles/commit/72120b7c4020017ed0479131ef0ddb7b5620d517))
* **LabelsExploration:** Introduce bar gauge visualisations ([#72](https://github.com/grafana/explore-profiles/issues/72)) ([7b1b19a](https://github.com/grafana/explore-profiles/commit/7b1b19a81e0ca6825bae9f2b06795199f4c9d209))
* **SceneLabelValuesTimeseries:** Colors and legends are preserved on expanded timeseries ([#85](https://github.com/grafana/explore-profiles/issues/85)) ([6980299](https://github.com/grafana/explore-profiles/commit/69802997b1a5fc72938bb0eaaf27e99076980f7a))
* Various enhancements after first UX interview ([#81](https://github.com/grafana/explore-profiles/issues/81)) ([2cdfcbe](https://github.com/grafana/explore-profiles/commit/2cdfcbecae5b1bd74310a3cbd8a115bc1e166525))



## [0.1.3](https://github.com/grafana/explore-profiles/compare/v0.1.2...v0.1.3) (2024-07-19)


### Bug Fixes

* **Header:** Switch the exploration type radio button group to a select on narrow screens ([#70](https://github.com/grafana/explore-profiles/issues/70)) ([55f420a](https://github.com/grafana/explore-profiles/commit/55f420a532ee8f2d6d955112d2dd4665df18cf67))



## [0.1.2](https://github.com/grafana/explore-profiles/compare/v0.0.46-explore-profiles-beta-35...v0.1.2) (2024-07-17)


### Bug Fixes

* **CompareAction:** Add missing data source query parameter to compare URL ([#58](https://github.com/grafana/explore-profiles/issues/58)) ([b1213e1](https://github.com/grafana/explore-profiles/commit/b1213e13aad71f11bbd8473571b4d9ae37924b8f))
* **FunctionDetails:** Get timeline state from Flame Graph component ([#25](https://github.com/grafana/explore-profiles/issues/25)) ([64ed0e6](https://github.com/grafana/explore-profiles/commit/64ed0e68a22445111b1d1ec02dff9b2fd8daecaa))
* **GitHub Integration:** Correctly extract the start/end timestamps from time picker ([#15](https://github.com/grafana/explore-profiles/issues/15)) ([fe8d807](https://github.com/grafana/explore-profiles/commit/fe8d807a83fce1b3b3b1eeb39d980af0312548bb))
* **SceneAllLabelValuesTableState:** Fix color contrast in light mode ([#26](https://github.com/grafana/explore-profiles/issues/26)) ([1bd268f](https://github.com/grafana/explore-profiles/commit/1bd268fd2bf2236ed9b6853e6d48a17933107bf5))
* **SceneByVariableRepeaterGrid:** Set timeseries min to 0 ([#31](https://github.com/grafana/explore-profiles/issues/31)) ([0e3a17d](https://github.com/grafana/explore-profiles/commit/0e3a17df3363cb2b61bab85039522e44eb766c61))
* **SceneFlameGraph:** Fix runtime error ([#45](https://github.com/grafana/explore-profiles/issues/45)) ([6227f2d](https://github.com/grafana/explore-profiles/commit/6227f2dcb1d705259fb1ad8ae9f144eb17cd80b1))
* **SceneFlameGraph:** Respect maxNodes when set in the URL ([#29](https://github.com/grafana/explore-profiles/issues/29)) ([85dd5b7](https://github.com/grafana/explore-profiles/commit/85dd5b79833f1737c0cf5505b743e50e256a20dc))


### Features

* **Analytics:** Track Explore Profiles actions ([#64](https://github.com/grafana/explore-profiles/issues/64)) ([ec58f57](https://github.com/grafana/explore-profiles/commit/ec58f5771c6ff59fcbd444ac62c2e55dd1bda202))
* **DataSource:** Store selected data source in local storage ([#60](https://github.com/grafana/explore-profiles/issues/60)) ([9f7ede1](https://github.com/grafana/explore-profiles/commit/9f7ede188279010502f2bcef02b2caba94b5064f))
* **SingleView:** Remove page ([#20](https://github.com/grafana/explore-profiles/issues/20)) ([16da70d](https://github.com/grafana/explore-profiles/commit/16da70d7f424c17982a8ca1ceab24a2589121007))
* Update plugin metadata to auto enable ([#65](https://github.com/grafana/explore-profiles/issues/65)) ([3afd1cd](https://github.com/grafana/explore-profiles/commit/3afd1cd6cbdaf93583978ecab80af8a620e313ef))
* Various minor improvements ([#46](https://github.com/grafana/explore-profiles/issues/46)) ([877b009](https://github.com/grafana/explore-profiles/commit/877b0094ffd21794b5742db6fbfb32ebd5868a4c))



# [0.1.0](https://github.com/grafana/explore-profiles/compare/v0.0.46-explore-profiles-beta-35...v0.1.0) (2024-07-15)

Explore Profiles is now available in its initial public release. It is designed to offer a seamless, query-less experience for browsing and analyzing profiling data.

Key features include:

- **Native integration with Pyroscope**: Seamlessly integrates with Pyroscope backend to provide detailed performance profiling and analysis.
- **Query-Less Browsing**: Navigate profiling data without the need for complex queries.
- **AI-Powered flame graph analysis**: uses a large-language model (LLM) to assist with flame graph data interpretation so you can identify bottlenecks, and get to the bottom of performance issues faster.

### Bug Fixes

- **GitHub Integration:** Correctly extract the start/end timestamps from time picker ([#15](https://github.com/grafana/explore-profiles/issues/15)) ([fe8d807](https://github.com/grafana/explore-profiles/commit/fe8d807a83fce1b3b3b1eeb39d980af0312548bb))
- **SceneAllLabelValuesTableState:** Fix color contrast in light mode ([#26](https://github.com/grafana/explore-profiles/issues/26)) ([1bd268f](https://github.com/grafana/explore-profiles/commit/1bd268fd2bf2236ed9b6853e6d48a17933107bf5))
- **SceneByVariableRepeaterGrid:** Set timeseries min to 0 ([#31](https://github.com/grafana/explore-profiles/issues/31)) ([0e3a17d](https://github.com/grafana/explore-profiles/commit/0e3a17df3363cb2b61bab85039522e44eb766c61))
- **SceneFlameGraph:** Fix runtime error ([#45](https://github.com/grafana/explore-profiles/issues/45)) ([6227f2d](https://github.com/grafana/explore-profiles/commit/6227f2dcb1d705259fb1ad8ae9f144eb17cd80b1))
- **SceneFlameGraph:** Respect maxNodes when set in the URL ([#29](https://github.com/grafana/explore-profiles/issues/29)) ([85dd5b7](https://github.com/grafana/explore-profiles/commit/85dd5b79833f1737c0cf5505b743e50e256a20dc))

### Features

- Bump version to 0.1.0 ([3e480f9](https://github.com/grafana/explore-profiles/commit/3e480f90c06cba6d9ac3558026a1c892963db4c6))
- **SingleView:** Remove page ([#20](https://github.com/grafana/explore-profiles/issues/20)) ([16da70d](https://github.com/grafana/explore-profiles/commit/16da70d7f424c17982a8ca1ceab24a2589121007))
- Various minor improvements ([#46](https://github.com/grafana/explore-profiles/issues/46)) ([877b009](https://github.com/grafana/explore-profiles/commit/877b0094ffd21794b5742db6fbfb32ebd5868a4c))
