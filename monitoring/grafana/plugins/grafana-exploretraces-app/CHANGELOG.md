# Changelog

## [1.0.0](https://github.com/grafana/traces-drilldown/compare/v0.2.9...v1.0.0) (2025-04-24)

* Breakdown: Do not show an empty state while streaming is still in progress. ([#426](https://github.com/grafana/traces-drilldown/pull/426))
* Add support for contextualised trace list table. ([#409](https://github.com/grafana/traces-drilldown/pull/409))
* Move version to menu and remove preview badge. ([#429](https://github.com/grafana/traces-drilldown/pull/429))
* Add fix to show empty state in the trace list. ([#430](https://github.com/grafana/traces-drilldown/pull/430))
* Fix to normalize comparison data when total fields are missing or invalid. ([#435](https://github.com/grafana/traces-drilldown/pull/435))
* Breakdown: create a new history item when a filter is added from the breakdown. ([#431](https://github.com/grafana/traces-drilldown/pull/431))

## [0.2.9](https://github.com/grafana/traces-drilldown/compare/v0.2.8...v0.2.9) (2025-04-15)

* Remove exemplars from heatmap. ([#398](https://github.com/grafana/traces-drilldown/pull/398))
* Filter out redundant attributes. ([#397](https://github.com/grafana/traces-drilldown/pull/397))
* Show warning if datasource is not configured with TraceQL metrics. ([#400](https://github.com/grafana/traces-drilldown/pull/400))
* Ensure Y-axis label matches the data for RED metrics. ([#401](https://github.com/grafana/traces-drilldown/pull/401))
* Explore: Add "Go Queryless" hook. ([#404](https://github.com/grafana/traces-drilldown/pull/404))
* Fix issue with container height. ([#422](https://github.com/grafana/traces-drilldown/pull/422))
* Use events to open traces. ([#410](https://github.com/grafana/traces-drilldown/pull/410))
* chore: remove extension link from logs drilldown. ([#421](https://github.com/grafana/traces-drilldown/pull/421))
* Fix structure tab flickering. ([#394](https://github.com/grafana/traces-drilldown/pull/394))
* Support typed query generation. ([#423](https://github.com/grafana/traces-drilldown/pull/423))
* RED Panels: update error panel y-axis labels. ([#424](https://github.com/grafana/traces-drilldown/pull/424))
* Rename plugin extension link from Explore to Drilldown. ([#425](https://github.com/grafana/traces-drilldown/pull/425))
* Add support for adding a trace to investigations. ([#408](https://github.com/grafana/traces-drilldown/pull/408))

## [0.2.6](https://github.com/grafana/traces-drilldown/compare/v0.2.4...v0.2.6) (2025-03-12)

### Enhancements

* Support for add to investigation. ([#320](https://github.com/grafana/traces-drilldown/pull/320))
* Support for metrics streaming. ([#312](https://github.com/grafana/traces-drilldown/pull/312))
* Rename plugin to Grafana Traces Drilldown. ([#329](https://github.com/grafana/traces-drilldown/pull/329))
* Add back and forward support for app actions. ([#294](https://github.com/grafana/traces-drilldown/pull/294))
* Exposes a component which takes properties and creates a LinkButton with a href to navigate to the Traces Drilldown from outside. ([#335](https://github.com/grafana/traces-drilldown/pull/335))
* Select custom columns in trace list. ([#342](https://github.com/grafana/traces-drilldown/pull/342))

## [0.2.3](https://github.com/grafana/explore-traces/compare/v0.2.2...v0.2.3) (2025-02-06)

### Enhancements

* **Open trace in drawer:** The traces now open in a drawer which should improve the experience of analysing the details of a trace. ([#325](https://github.com/grafana/explore-traces/pull/325))

### Bug Fixes

* Fixes crash on main metric panel ([#317](https://github.com/grafana/explore-traces/pull/317))

## [0.2.2](https://github.com/grafana/explore-traces/compare/v0.2.0...v0.2.2) (2025-01-13)

### Enhancements

* **Custom values in filters bar:** The filters bar now allows custom values which can be used to build regular expressions or input values missing from the dropdown options. ([#288](https://github.com/grafana/explore-traces/pull/252))

## [0.2.0](https://github.com/grafana/explore-traces/compare/v0.1.3...v0.2.0) (2025-01-10)

### Features

* **Support for exemplars:** Quickly jump to the relevant data points or logs for deeper troubleshooting with newly added support for exemplars, directly on your metrics graph. By clicking on a point of interest on the graph—like a spike or anomaly—you can quickly jump to the relevant traces for deeper troubleshooting and dramatically reduce the time it takes to root cause an issue. ([#278](https://github.com/grafana/explore-traces/pull/278)) Requires Grafana >= 11.5.0
* **Open traces in Explore:** When viewing trace spans, now you can easily open the full trace in Explore. This provides a streamlined way to pivot between trace analysis and the broader Grafana Explore experience without losing context. ([#267](https://github.com/grafana/explore-traces/pull/267))

### Enhancements

* **Trace breakdown adjusts better to smaller screens:** The **Breakdown** tab now automatically adjusts its attribute selector display based on available screen width, improving usability on smaller viewports. ([#267](https://github.com/grafana/explore-traces/pull/267))
* **Search is now case-insensitive:** Search in the **Breakdown** and **Comparison** tabs now ignores capitalization, ensuring you see all matching results. ([#252](https://github.com/grafana/explore-traces/pull/252))
* **Performance boost and reduced bundle size**: Code-splitting and lazy loading for faster loading times. Only the modules you need are fetched on demand, cutting down on initial JavaScript payload and improving app performance. ([#275](https://github.com/grafana/explore-traces/pull/275))
* **Various fixes and improvements:** Fixed loading and empty states. Fixed broken documentation link. Refined styles above filters for a more polished look. Added descriptive text to the Span List tab for added clarity. Enhanced tooltip design for RED metrics. Standardized error messages and titles, plus added helpful hints when an empty state appears. ([#263](https://github.com/grafana/explore-traces/pull/263))

## 0.1.2

Release public preview version.
