const total = surveyResponses.length;

const promoters =
surveyResponses.filter(x => x.score >= 9).length;

const passives =
surveyResponses.filter(
 x => x.score >= 7 && x.score <= 8
).length;

const detractors =
surveyResponses.filter(x => x.score <= 6).length;

console.table({
 total,
 promoters,
 passives,
 detractors,
 sum:
 promoters + passives + detractors
});