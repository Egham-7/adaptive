/* [create-plugin] version: 5.11.1 */
define([
	"@emotion/css",
	"@grafana/data",
	"@grafana/runtime",
	"@grafana/ui",
	"d3",
	"lodash",
	"module",
	"react",
	"react-dom",
	"react-router",
	"rxjs",
], (e, r, o, a, t, n, i, p, s, l, c) =>
	(() => {
		let u;
		let d;
		const f = {
			6089: (r) => {
				r.exports = e;
			},
			7781: (e) => {
				e.exports = r;
			},
			8531: (e) => {
				e.exports = o;
			},
			2007: (e) => {
				e.exports = a;
			},
			4201: (e) => {
				e.exports = t;
			},
			3241: (e) => {
				e.exports = n;
			},
			1308: (e) => {
				e.exports = i;
			},
			5959: (e) => {
				e.exports = p;
			},
			8398: (e) => {
				e.exports = s;
			},
			1159: (e) => {
				e.exports = l;
			},
			1269: (e) => {
				e.exports = c;
			},
		};
		const g = {};
		function m(e) {
			const r = g[e];
			if (void 0 !== r) return r.exports;
			const o = (g[e] = { id: e, loaded: !1, exports: {} });
			return f[e].call(o.exports, o, o.exports, m), (o.loaded = !0), o.exports;
		}
		(m.m = f),
			(m.amdO = {}),
			(m.n = (e) => {
				const r = e?.__esModule ? () => e.default : () => e;
				return m.d(r, { a: r }), r;
			}),
			(m.d = (e, r) => {
				for (const o in r)
					m.o(r, o) &&
						!m.o(e, o) &&
						Object.defineProperty(e, o, { enumerable: !0, get: r[o] });
			}),
			(m.f = {}),
			(m.e = (e) =>
				Promise.all(Object.keys(m.f).reduce((r, o) => (m.f[o](e, r), r), []))),
			(m.u = (e) => `${e}.js`),
			(m.g = (function () {
				if ("object" === typeof globalThis) return globalThis;
				try {
					return this || new Function("return this")();
				} catch (e) {
					if ("object" === typeof window) return window;
				}
			})()),
			(m.o = (e, r) => Object.prototype.hasOwnProperty.call(e, r)),
			(u = {}),
			(d = "grafana-pyroscope-app:"),
			(m.l = (e, r, o, a) => {
				if (u[e]) u[e].push(r);
				else {
					let t;
					let n;
					if (void 0 !== o)
						for (
							let i = document.getElementsByTagName("script"), p = 0;
							p < i.length;
							p++
						) {
							const s = i[p];
							if (
								s.getAttribute("src") === e ||
								s.getAttribute("data-webpack") === d + o
							) {
								t = s;
								break;
							}
						}
					t ||
						((n = !0),
						((t = document.createElement("script")).charset = "utf-8"),
						(t.timeout = 120),
						m.nc && t.setAttribute("nonce", m.nc),
						t.setAttribute("data-webpack", d + o),
						(t.src = e),
						0 !== t.src.indexOf(`${window.location.origin}/`) &&
							(t.crossOrigin = "anonymous"),
						(t.integrity = m.sriHashes[a]),
						(t.crossOrigin = "anonymous")),
						(u[e] = [r]);
					const l = (r, o) => {
						(t.onerror = t.onload = null), clearTimeout(c);
						const a = u[e];
						if (
							(delete u[e],
							t.parentNode?.removeChild(t),
							a?.forEach((e) => e(o)),
							r)
						)
							return r(o);
					};
					const c = setTimeout(
						l.bind(null, void 0, { type: "timeout", target: t }),
						12e4,
					);
					(t.onerror = l.bind(null, t.onerror)),
						(t.onload = l.bind(null, t.onload)),
						n && document.head.appendChild(t);
				}
			}),
			(m.r = (e) => {
				"undefined" !== typeof Symbol &&
					Symbol.toStringTag &&
					Object.defineProperty(e, Symbol.toStringTag, { value: "Module" }),
					Object.defineProperty(e, "__esModule", { value: !0 });
			}),
			(m.nmd = (e) => ((e.paths = []), e.children || (e.children = []), e)),
			(m.p = "public/plugins/grafana-pyroscope-app/"),
			(m.sriHashes = {
				36: "sha256-jHR/yzlFemrvQ7Rv/O2DRqMs2A2qgk9uWKt5jfLAuyU=",
				350: "sha256-98kzEJz25Ct3FGMJ+jPD0mZFsLwt5s2PI+raKCvX2bs=",
			}),
			(() => {
				const e = { 231: 0 };
				m.f.j = (r, o) => {
					let a = m.o(e, r) ? e[r] : void 0;
					if (0 !== a)
						if (a) o.push(a[2]);
						else {
							const t = new Promise((o, t) => (a = e[r] = [o, t]));
							o.push((a[2] = t));
							const n = m.p + m.u(r);
							const i = new Error();
							m.l(
								n,
								(o) => {
									if (m.o(e, r) && (0 !== (a = e[r]) && (e[r] = void 0), a)) {
										const t = o && ("load" === o.type ? "missing" : o.type);
										const n = o?.target?.src;
										(i.message = `Loading chunk ${r} failed.\n(${t}: ${n})`),
											(i.name = "ChunkLoadError"),
											(i.type = t),
											(i.request = n),
											a[1](i);
									}
								},
								`chunk-${r}`,
								r,
							);
						}
				};
				const r = (r, o) => {
					let a;
					let t;
					const [n, i, p] = o;
					let s = 0;
					if (n.some((r) => 0 !== e[r])) {
						for (a in i) m.o(i, a) && (m.m[a] = i[a]);
						if (p) p(m);
					}
					for (r?.(o); s < n.length; s++)
						(t = n[s]), m.o(e, t) && e[t] && e[t][0](), (e[t] = 0);
				};
				const o = (self.webpackChunkgrafana_pyroscope_app =
					self.webpackChunkgrafana_pyroscope_app || []);
				o.forEach(r.bind(null, 0)), (o.push = r.bind(null, o.push.bind(o)));
			})();
		const y = {};
		m.r(y), m.d(y, { plugin: () => $ });
		const v = m(1308);
		const h = m.n(v);
		m.p = h()?.uri
			? h().uri.slice(0, h().uri.lastIndexOf("/") + 1)
			: "public/plugins/grafana-pyroscope-app/";
		const b = m(7781);
		const x = m(5959);
		const w = m.n(x);
		const S = w().lazy(() =>
			Promise.all([m.e(350), m.e(36)])
				.then(m.bind(m, 4763))
				.then((e) => ({ default: e.App })),
		);
		function T(e) {
			let r;
			let o;
			let a;
			const { timeRange: t, pyroscopeQuery: n } = e;
			let i = "";
			let p = "";
			let s = "all";
			const l =
				null === (o = e.pyroscopeQuery.labelSelector) ||
				void 0 === o ||
				null === (r = o.match(/service_name="([^"]+)"/)) ||
				void 0 === r
					? void 0
					: r[1];
			l && (s = "labels"), e.explorationType && (s = e.explorationType);
			const c = `var-dataSource=${null === (a = n.datasource) || void 0 === a ? void 0 : a.uid}`;
			const u = l ? `&var-serviceName=${l}` : "";
			const d = `&var-profileMetricId=${n.profileTypeId}`;
			const f = `&explorationType=${s}`;
			t && (i = `&from=${t.from}&to=${t.to}`),
				n.spanSelector && (p = `&var-spanSelector=${n.spanSelector}`);
			return `/a/grafana-pyroscope-app/explore?${new URLSearchParams(`${c}${u}${d}${i}${f}${p}`).toString()}`;
		}
		const k = {
			targets: [b.PluginExtensionPoints.ExploreToolbarAction],
			title: "Open in Grafana Profiles Drilldown",
			icon: "fire",
			description: "Try our new queryless experience for profiles",
			path: "/a/grafana-pyroscope-app/explore",
			configure(e) {
				if (!e || !e.targets || !e.timeRange || e.targets.length > 1) return;
				const r = e.targets[0];
				return r.datasource &&
					"grafana-pyroscope-datasource" === r.datasource.type
					? { path: T({ pyroscopeQuery: r, timeRange: e.timeRange }) }
					: void 0;
			},
		};
		const O = {
			targets: ["grafana/traceview/details"],
			title: "Open in Grafana Profiles Drilldown",
			description: "Try our new queryless experience for profiles",
			path: "/a/grafana-pyroscope-app/explore",
			onClick: (e, { context: r }) => {
				if (
					!(r?.serviceName && r.spanSelector && r.profileTypeId && r.timeRange)
				)
					return;
				const o = r.serviceName;
				const a = r.spanSelector;
				const t = r.profileTypeId;
				const n = r.timeRange;
				const i = {
					refId: "span-flamegraph-profiles-drilldown-refId",
					labelSelector: `service_name="${o}"`,
					profileTypeId: t,
					spanSelector: a,
					datasource: r.datasource,
					groupBy: ["service_name"],
				};
				if (i.datasource) {
					const e = T({
						pyroscopeQuery: i,
						timeRange: n,
						explorationType: "flame-graph",
					});
					window.open(e, "_blank", "noopener,noreferrer");
				}
			},
		};
		const $ = new b.AppPlugin()
			.addLink(k)
			.addLink(O)
			.setRootPage(() =>
				w().createElement(x.Suspense, null, w().createElement(S, null)),
			);
		return y;
	})());
//# sourceMappingURL=module.js.map
