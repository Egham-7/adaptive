export const createCacheHeaders = () => {
	const ONE_MINUTE = 60;
	const ONE_HOUR = 3600;

	return new Headers([
		[
			"cache-control",
			`s-maxage=${ONE_MINUTE}, stale-while-revalidate=${ONE_HOUR}`,
		],
		["vary", "authorization, cookie"],
	]);
};

export const createCacheInvalidationHeaders = (tags: string[]) => {
	const headers = new Headers([
		["cache-control", "no-cache, no-store, must-revalidate"],
		["pragma", "no-cache"],
		["expires", "0"],
	]);

	if (tags.length > 0) {
		headers.set("cache-invalidate", tags.join(","));
	}

	return headers;
};
