import { useCallback, useEffect, useRef, useState } from "react";

// How many pixels from the bottom of the container to enable auto-scroll
const ACTIVATION_THRESHOLD = 50;
// Minimum pixels of scroll-up movement required to disable auto-scroll
const MIN_SCROLL_UP_THRESHOLD = 10;

export function useAutoScroll(
	dependencies: React.DependencyList,
	isStreaming?: boolean,
) {
	const containerRef = useRef<HTMLDivElement | null>(null);
	const previousScrollTop = useRef<number | null>(null);
	const [shouldAutoScroll, setShouldAutoScroll] = useState(true);
	const scrollAnimationFrameRef = useRef<number | null>(null);

	const scrollToBottom = useCallback(() => {
		if (containerRef.current) {
			containerRef.current.scrollTo({
				top: containerRef.current.scrollHeight,
				behavior: "smooth",
			});
		}
	}, []);

	const scrollToBottomSmooth = useCallback(() => {
		if (containerRef.current && shouldAutoScroll) {
			const container = containerRef.current;

			// Only auto-scroll if we're close to the bottom already
			const distanceFromBottom = Math.abs(
				container.scrollHeight - container.scrollTop - container.clientHeight,
			);
			if (distanceFromBottom < ACTIVATION_THRESHOLD * 2) {
				// Use instant scroll during streaming for better performance
				container.scrollTop = container.scrollHeight;
			}
		}
	}, [shouldAutoScroll]);

	const handleScroll = () => {
		if (containerRef.current) {
			const { scrollTop, scrollHeight, clientHeight } = containerRef.current;

			const distanceFromBottom = Math.abs(
				scrollHeight - scrollTop - clientHeight,
			);

			const isScrollingUp = previousScrollTop.current
				? scrollTop < previousScrollTop.current
				: false;

			const scrollUpDistance = previousScrollTop.current
				? previousScrollTop.current - scrollTop
				: 0;

			const isDeliberateScrollUp =
				isScrollingUp && scrollUpDistance > MIN_SCROLL_UP_THRESHOLD;

			if (isDeliberateScrollUp) {
				setShouldAutoScroll(false);
			} else {
				const isScrolledToBottom = distanceFromBottom < ACTIVATION_THRESHOLD;
				setShouldAutoScroll(isScrolledToBottom);
			}

			previousScrollTop.current = scrollTop;
		}
	};

	const handleTouchStart = () => {
		setShouldAutoScroll(false);
	};

	useEffect(() => {
		if (containerRef.current) {
			previousScrollTop.current = containerRef.current.scrollTop;
		}
	}, []);

	// Effect for regular message updates
	useEffect(() => {
		if (shouldAutoScroll) {
			// Use instant scroll for new messages to avoid lag
			if (containerRef.current) {
				containerRef.current.scrollTop = containerRef.current.scrollHeight;
			}
		}
	}, [shouldAutoScroll, ...dependencies]);

	// Effect for streaming content - continuously scroll during streaming
	useEffect(() => {
		if (isStreaming && shouldAutoScroll) {
			const scheduleScroll = () => {
				scrollToBottomSmooth();
				scrollAnimationFrameRef.current = requestAnimationFrame(scheduleScroll);
			};

			scrollAnimationFrameRef.current = requestAnimationFrame(scheduleScroll);

			return () => {
				if (scrollAnimationFrameRef.current) {
					cancelAnimationFrame(scrollAnimationFrameRef.current);
					scrollAnimationFrameRef.current = null;
				}
			};
		}
	}, [isStreaming, shouldAutoScroll, scrollToBottomSmooth]);

	return {
		containerRef,
		scrollToBottom,
		handleScroll,
		shouldAutoScroll,
		handleTouchStart,
	};
}
