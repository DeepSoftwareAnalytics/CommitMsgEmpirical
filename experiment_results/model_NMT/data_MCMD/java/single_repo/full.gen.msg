Minor DocWriteResponse changes ( )
Merge pull request elastic / elasticsearch from rjernst / rest_alone
[ TEST ] Enhance SimpleQueryStringIT # testConcurrentJoining ( ) ( )
Core : Add synthetic checksum to the DeleteAlertResponse
Remove [ Transform ] docs
Adjust CoordinatorTests ( )
Packaging : Fix randomizedtest fail
Security : mention if version is disabled for searchable snapshots ( )
Merge pull request from jimferenczi / max_merge_settings
Support WKT sentences docs in docs ( )
Change BlobPath . buildAsString ( ) to 500 ( RestStatus . INTERNAL_SERVER_ERROR )
Removes distribution code for xpack src
review comments
Mappings : Remove Some more of
Add async_search failures to timestamp field ( )
Add link to release notes
Fail a source if it is used to Lucene ( )
[ query ] Add support for " authorization_realms " to " parsing " .
Update to Lucene 4 . 10 . 2
removed useless committed test index . mapping .
[ TEST ] add afterDistribution timeout to expectedTimeToHeal
Removed unused code .
Merge pull request from jasontedor / remove - CONF_FILE
Add UBI docker builds ( )
Add integration tests for index ( )
[ DOCS ] Add data streams to cluster reroute API docs ( )
Merge pull request from tebriel / plugin_docs
[ BUILD ] Use SuppressFrobidden annotation instead of class level excludes
SQL : MATCH and QUERY documentation ; one list of functions ( )
upgrade to lucene 3 . 0 . 0 - snapshot - 08b8d116f8f .
Do not block Translog add on file write ( )
Remove dead leniency
test : InternalEngineTests . testSegmentsWithMergeFlag should close it', "'", 's translog
fix match_all type for match_all filter
Reset hunspell inputstream to all characters
Merge pull request from rjernst / buildsrc_test_cleanup
More AbstractIntegrationTest
Add unit tests for checkPart1 ( )
Merge pull request from rjernst / deprecate_exception_hurter
check also on the node settings
Merge pull request elastic / elasticsearch from rjernst / rest_alone
Fix InternalSearchHit # hasSource to return a proper for a default
Merge pull request from s1monw / fold_in_index_cache_module
Rename AggregatorBuilder to writeGenericValue
Default changes to node . settings ( )
Merge pull request from ywelsch / fix / handling_index_state
Adding core
Merge pull request from ywelsch / fix / index - cs - restore
Merge pull request from rmuir / eight_point_three
Merge pull request from jasontedor / remove - CONF_FILE
Cluster State : Remove index template filtering
Add index poll_interval to breaking changes
Correct reindex stats for field_caps tests ( )
fix asciidoc syntax to docs when running processor ( )
Wire up IpRangeAggregation to ValuesSourceRegistry ( )
Docs : CONSOLEify over docs . ( )
SQL : Fix issue with WEEK / ISO_WEEK / DATEDIFF ( elastic / x - pack - elasticsearch )
painless : Decouple PainlessLookupBuilder . mail should be set
Depend on elasticsearch - parent
Internal : Remove mapper listeners
Dependencies : Update joda - time .
Merge pull request elastic / elasticsearch from PhaedrusTheGreek / simplify_ssl_test
Internal : Remove dead code from mapping , use it in favor of the delay directly
Fix elasticsearch on OSX
reenable bwc test now that is merged ( )
[ DOCS ] EQL : Remove case sensitivity from function docs ( )
[ DOCS ] Fine - tunes training_percent definition . ( )
LLRC : add DateFormatters and DeadHostState
Remove unused imports from server ( )
[ TEST ] fix IndexAliasesTests
Upgrade gradle to 6 . 1 . 1 ( )
Remove SecurityTransportModule
Add version for
[ DOCS ] Update monitoring to cluster - pending
[ DOCS ] Reformats nodes for index settings API ( )
Fixed typo
Plugins : Fix plugin - test configuration for integ tests
Update indices . asciidoc

Remove unused code
[ CORE ] Ban duplicate useage of Future # cancel ( true )
Watcher : Fix issue in MS query ( )
Cherry pick tests and seqNo recovery hardning ( )
[ DOCS ] Fix X - Pack references ( elastic / x - pack - elasticsearch )
Native . lastSeenAccessTime should be final
remove stale settings for shield
Mute AsyncSearchActionTests ( )
only use a tiny thread pool
Build : Fix plugin issue in test build ( )
[ DOCS ] Reformat ` search ` token filter ( )
Added a debug logging to rest APIs
Derivative and CircuitBreakerStats to implement Writeable rather than Streamable
Merge pull request from rmuir / allow_master_elect_extension
[ ML ] Mute data frame analytics BWC tests ( )
Test : crank up tests
Use security when role is simple
Adjust error in hidden errmsg ( )
Split BWC tests for rest - 5 ( )
doc
fix test
[ DOCS ] Fixed xpack . watcher . cleaner_service_enabled
Remove references to mapping type in FieldTypeLookup . ( )
Clarified note about needing to perform a full cluster restart . Closes elastic / elasticsearch .
Removes unused code
Add support for the absolute interval interval to the HLRC ( )
fix log
[ ML ] [ Inference ] Add integ tests
Fix SearchableSnapshotActionIT ( )
Add missing rest test for search settings ( )
[ TEST ] Don', "'", 't use acks when isolating nodes
Fix a typo in ActionFilter
[ TEST ] Fix test bug in MultiOrdinalsTests .
security : Replace ` _shield ` and ` max_value_as_string ` in shield
[ ML ] Remove unused Request constructors ( )
minor cleanups
Merge pull request from rmuir / rat
Fix compilation issue on node assignment in lambda stats
apply review comments
[ DOCS ] Add xref to multiplexer suggest
[ ML ] Mute data frame analytics BWC tests ( )
Data stream support for migrations deprecations info ( )
Handle SearchContext on the absolute ( )
Engine : store maxUnsafeAutoIdTimestamp in commit ( )
body name was not required in the cluster state .
Added ShapeFetchService for more tests
[ ML ] Do not don', "'", 't shard stopping when creating a job ( )
Add async_search ( ) method ( )
[ Monitoring ] Add create_follow_index
Add set setting to tasks
[ TEST ] add a lot of forgotten null
make sure we use proper default settings
protected when not of the same type .
[ ML ] Remove partition_field interface ( )
Merge pull request from rjernst / client_jar_id
Unmute BasicLicenseUpgradeIT ( )
Query refactoring : SpanNearQueryBuilder and Parser
Docs : add Bloodhound
Docs : CONSOLEify task_id . Closes ( elastic / elasticsearch )

Merge pull request elastic / elasticsearch from rjernst / rest_alone
fix RandomScoreFunctionTests
Build : Fix license detection in Eclipse ( )
HTTP : fix wrong of of packages
Remove unused code
Add fromXContent methods to SuggestBuilder
Move the number of terms queries in InnerHitBuilder .
Merge pull request from elastic / fix / indices - get - restspec
monitoring ui : add heap . size_limit to Kibana stats
Scripting : Remove marvel_ transport master order .
add missing index to plugins timeout in rest tests ( )
Fix randomized test failure ( )
Revert " Fixed 5 . 0 . 0 - alpha2
Mute failing tests ( )
Remove support for onModule ( )
Merge pull request elastic / elasticsearch from dakrone / only - one - extension
Fix wording in RangeAggregator
Convert more sum fields to query docs ( )
[ ML ] [ Inference ] Unify top_classes object field names ( )
Merge pull request elastic / elasticsearch from rjernst / rest_alone
EQL : Introduce IsNull time to simplify expressions ( )
harden logging in rolling upgrade tests ( )
add more roles ( )
SQL : cover coercion in JdbcResultSet to throw SQLException ( elastic / x - pack - elasticsearch )
[ TEST ] remove duplicated ClusterScope
Use terms instead of ConstantScoreQuery in terms query
internal custom snapshot commands
Merge pull request elastic / elasticsearch from PhaedrusTheGreek / group_search_noattrs
XContentBuilder to implement 0 . 90 . 1
Test : ClusterServiceTetsts . testLocalNodeMasterListenerCallbacks should be null if the same alert is internal
Merge pull request from ywelsch / fix / index - delete - port
Fix rest - cli - multi - query test ( )
move
Remove ParseFieldMatcher as script fields from the ingest pipeline ( )
Also', handle 'InternalPolygonBuilder
Remove SecurityTransportModule
[ DOCS ] Document ` size ` API ( )
[ DOCS ] Fix edge n - gram tokenizer nav
Fix compilation issue with final assignment in lambda on windows jdk
Test : finish an typo in the error message
Merge pull request from cbuescher / feature / query - refactoring - constantscore
[ ML ] Fix online doc names in _explain ( )
SQL : Fix test for UnresolvedRelation
refactor indices sig
Internal : Deguice RepositoriesService ( )
Remove SecurityTransportModule
Watcher : Hide deactivated action with watcher action in CrossClusterSearch
add a debug logging to CONTRIBUTING . md to 2 . x ( )
fix build
fix test
Adjust BWC for 50836 ( )
Fix test stability
IndexNameExpressionResolver should not fail when closing an index
Mute upgrade tests ( )
[ TEST ] Use security . initial_master_nodes ( elastic / x - pack - elasticsearch )
Merge pull request from rmuir / harden_integration_tests
Merge pull request from rjernst / simplify_bwc_path
Removes unused imports
Merge pull request elastic / elasticsearch from jasontedor / sets - be - gone
Added spelling
Merge pull request from kkirsche / patch - 1
Provide error message when transport is unavailable ( )
Remove javax . Request ( ) ( )
change rollover rest - api placeholder name
[ DOCS ] Re - enabled ]', 1 'to control in Elasticsearch page
Upgrade to snapshot tasks with value ( )
Merge pull request from rmuir / eight_point_three
Test : More attempts at forcing netty test to always use netty
Remove cruft
Merge pull request from rjernst / client_jar_id
Rename the policies in the rest index
remove flushing from cluster state names test ( )
CCR : Requires soft - deletes on the follower ( )
don', "'", 't wait for all mappings within the cluster state .
Remove heuristics for cluster . 8 ( )
prepare for next development iteration
Merge pull request from ywelsch / fix / block - delete - port
Merge pull request from jpountz / upgrade / lucene - 5 . 2 . 0 - snapshot - 1673124
Added " lifecycle " and cleanups
Avoid circular reference in exception
[ TEST ]', added 'ordering of simple files
Notify default value
Merge pull request from rmuir / shave_mustache
Aggregations : Add cumulative sum aggregation
improve spelling
[ TEST ] Add missing debug output to HasChildQueryBuilderTests
Clear Gateway : Add ` reverse_cout ` , and ` reverse_term ` to ` order ` , closes .
Tests : Fix systemkeytool tests
[ DOCS ] Document ` xpack . http . proxy . scheme ` setting ( )
Watcher : Remove', the 'debug cache for the job ( elastic / x - pack - elasticsearch )
Use Lucenes expectThrows ( )
Fix testing on Windows log cacheability ( )
upgrade to lucene 5 . 0 . 0 - snapshot - 08b8d116f8f ( )
Test : An the ` percolator ` parameter to ` reverse_term ` . ( elastic / x - pack - elasticsearch )
Fix IndexShardTests # testWithRandomException ( )
ingest : fix BucketAggScript ( )
[ TEST ] Pass SearchHitsTests to AbstractStreamableXContentTestCase ( )
Upgrade to Lucene 4 . 8 . 2
Add Assertions to registration
[ DOCS ] Fix anchor syntax
[ TEST ] Mute IndicesCustomDataPathTests
Scripting : Fix IndexRecoveryCollectorTests
Manually synchronize listeners while iterating on them ( )
Buffer with ` beforeIndexAddedToCluster ` updateable in the english code .
Enable retention lease integration tests ( elastic / x - pack - elasticsearch )
Add System . 2 . 1 version
Upgrade to snapshot Lucene v 8 . 5
Remove options from Watcher support ( elastic / x - pack - elasticsearch )
Added 0 - info typo
[ TEST ] Remove needless ClusterScope annotation from NettyTransportMultiPortTests
[ DOCS ] Fix typo ( elastic / x - pack - elasticsearch )
[ Docs ] Add missing word in MlStrings . sh .
Remove references to watcher rest tests
The name parameter in indices . get_template is not required .
Data stream support for migrations bwc tests ( )
Merge pull request from rmuir / shave_mustache
internal Failure in get index api api
[ DOCS ] Fix typos in jump link . ( )
Merge pull request elastic / elasticsearch from rjernst / move_license
[ TEST ] Randomize FS ssl time
more the bind the cluster state based on the master , closes .
Build : Switch to new plugin configuration for integ tests
Add Failure to the mappings ( )
[ STORE ] Calculate Alder32 Checksums for legacy files in Store # checkIntegrity
Merge pull request from s1monw / renew_syn_on_merge
No the watch service
lucene 4 : Upgraded ContentIndexSearcher
[ TEST ] Fix timeout issue in PluginCliTests on Windows
Mute failing API ( )
Enable Gradle ( )
[ Docs ] Update Copyright notices in 2018 ( )
Spelling : Add ` query_cache ` option to HLRC ( )
Fix tests for now
Docs : Update the', Kafka 'plugin asciidoc
[ DOCS ] Add note to cluster . max_buckets ( )
Rename CCR - java - agent ( )
lucene 4 : Upgraded ContentIndexSearcher
Merge pull request from jdconrad / stype
add settings to the new scheme
allow to set the cluster . . .
Merge pull request from droberts195 / grep_minus_e_portability_fix
[ TEST ] use assertAcked ( ) calls to immutableCluster ( ) . size
Add custom link
Merge pull request from spinscale / 1503 - cleanup - remove - bytesarray - unsafe
Upgrade to lucene - 8 . 5 . 0 - snapshot - b313618cc1d ( )
Improve checks for assertions on master changes ( )
Added 2 . 3 . 9 - SNAPSHOT and 1 . 4 . 4 - SNAPSHOT versions
fixed test bug
[ TEST ] Randomize FS Translog implementation
Docs : Fixed mistake in the docs
add the {
Use hasValue ( ) ( )
Merge pull request elastic / elasticsearch from rmuir / simplify_ssl_test
Remove SearchRequestParsers ( elastic / elasticsearch )
Merge pull request from rjernst / buildsrc_test_cleanup
Testclusters : fix for integ tests ( )
Upgrade to Gradle 6 . 2 . 2 ( )
Adds docs for terms aggregation in the percolate aggregation .
Updated the TreeMap in the rest api spec
review comments
[ TEST ] remove AwaitsFix in testSnapshotSucceedsAfterSnapshotFailure ,
inner_hits : Remove marvel_
[ DOCS ] Fix typo ( ) ( )
Merge remote - tracking branch', "'", 'upstream / master', "'
[ DOCS ] Add ML analytical functions ( elastic / x - pack - elasticsearch )
Fix TcpTransport grammar
Transport : An analyzer mapper allowing to control the index analyzer of a document based on a document field , closes .
Merge pull request from rmuir / painless_compile_exceptions
HLRest : add security authenticate API ( )
[ DOCS ] Tidied up anchors after a non - default indices ( )
[ TEST ] Randomize FS * time
Remove buildResources from ScriptService
[ DOCS ] Clarify phrase query context smoothing parameter ( )
Testclusters : fix left ( )
Add ` beforeIndexAddedToCluster ` callback
Support for polish_stem field
[ DOCS ] Add xref to cluster - level . asciidoc
Don', "'", 't detect IndexShard # index calls ( )
Merge pull request from rmuir / crazy_ubuntu
Mute failing tests in tests
Merge pull request from MaineC / bugfix / movavg_double_predict
Upgrade to lucene - 8 . 0 . 0 - snapshot - 3333ce7da6d ( )
[ DOCS ] Add links to Elasticsearch book ( elastic / elasticsearch )
SQL : Relax test handling of CAST and CONVERT ( )
Fix version compatability test
[ DOCS ] Add missing anchors for security security docs ( )
Merge pull request from rmuir / client_jar_id
[ TEST ] fix explain method in BulkRequestModifierTests
[ DOCS ] Merges role ordering for transform API docs ( )
Mapper : crank up PainlessMethod ( elastic / x - pack - elasticsearch )
painless : add trap tests
[ ML ] Include assertion in field_stats integ test test ( )
Remove InternalTestCluster . fullRestart ( )
Transport : Add DateFormatters class ( )
[ CORE ] ElasticsearchIntegrationTest . clearDisruptionScheme should also if it is applied
Fix TcpTransport + tests ( )
[ DOCS ] Add docs for get get API docs ( )
Add test for wrong logging in mixed cluster tests ( )
[ ML ] Logs error when DFA task is set to failed ( )
Plugins : Clean up PainlessMethod ( )
Merge pull request from cbuescher / introduce - shapebuilders
add boost_mode to registration
Fix SearchableSnapshotActionIT ( )
SQL : Polish IsNull About column expression ( )
Docs : Fix mistake in percolate_format docs
Security improvements
Add ` ensureGreen ` to index ( )
` sorting ` is not', supported 'in SearchSourceBuilder
security : make sure that watcher action is created ( elastic / elasticsearch )
Build : Fix compilation check for timevalue in Gradle ( )
Fix testRetentionLeasesClearedOnRestore ( )
[ DOCS ] Fixes typo in jump link ( )
Merge pull request from rjernst / deprecate_exception_hurter
[ TEST ] remove SecurityClusterClientYamlTestCase ( )
EQL : Introduce support for sequences ( )
first cut into toQuery and formXContent
Add fixture support for repository - s3
Add built - in -', Add 'from config ( )
Implement ctx . asciidoc ( )
Switching HighlighterBuilder for ip_range aggregation . ( )
Migrate TEST to REST tests ( )
[ ML ] Remove JobTaskStatus # staleTask ( . . . ) method
Merge pull request from rjernst / deprecate_exception_hurter
[ DOCS ] Changes experimental flag to beta in DFA related docs ( )
Mute test SLMSnapshotBlockingIntegTests . testRetentionWhileSnapshotInProgress
Remove PreProcessModule
Sorting on _score in the URI format , closes .
[ DOCS ] Fixes broken stack detection in stack , no own ,', fix '( )
[ TEST ] Actually perform the intended close . . .
[ TEST ] fix CommonTermsQueryBuilderTest
Merge remote - tracking branch', "'", 'es / master', "'", 'into feature / ingest
Upgrade to snapshot Lucene v 8 . 5
Fix perf bug in Painless tests
Add Assertions about BytesReference ( )
Add a comment to MapperBuilders ( )
fix top_terms_boost example
Fix LoggingOutputStream in NodeConnectionsServiceTests ( )
bring back the content parser , don', "'", 't create it , closes .
[ ML ] Hide partition_scores the datafeed lifecycle management
Log logic of CCR errmsg ( )
improve code
Merge pull request from jasontedor / feature / cleanup
Switch AggregationSpec to ContextParser ( )
Change DataFrameAnalyticsAuditMessage # testForgetFollower
Don', "'", 't recursively count', "'", 's ` field . type ( ) .
Mute failing test ( )
[ DOCS ] Fix typos
[ DOCS ] Fix typo in jump link . ( )
Build : Add fixture capabilities to integ tests
Fix docs for cluster reroute docs
[ REST : Add index action to index ( )
SCRIPTING : Support BucketAggScript ( )
Merge pull request from rjernst / buildsrc_test_cleanup
Remove deprecated modifier
Merge pull request from mikemccand / close_tokenstream
[ ML ] MlMemoryTracker should ignore analytics tasks without analytics ( )
Only use Optional in Sniffer
[ nodes ] unmute FullClusterRestartIT . testShrink test
Remove types for templates ( )
[ DOCS ] Changes - attachment configuring pages ( )
Added a {
Depend on elasticsearch - parent
moved AwaitsFix to unit test
Fix line issue in searchable snapshots tests ( )
Don', "'", 't assume prepare index request
Now with the new snapshot of an empty version ( )
Update similarity . asciidoc
Fix bwc tests after build of master
[ PARSER ] [ 1 . 0 ] version
[ TEST ] use assertAcked ( ) calls to expectedTimeToHeal
[ Security ] [ Inference ] Unify doSaveState with optimistic array ( )
[ DOCS ] Add xref to multiplexer . asciidoc ( )
Use ClusterState . buildAsString ( ) to Shard $ 1ClusterStateTaskExecutor
add basic script services
Merge pull request from s1monw / remove_bloom
SQL : Fix column size for SYS TYPE ( elastic / x - pack - elasticsearch )
Move AggregationSpec to ContextParser ( )
copy Actions to failNoIndices ( )
[ DOCS ] Add links to Beats security pages ( elastic / x - pack - elasticsearch )
fix sample docs for minhash .
improved settings
Remove nio 4 . 8 . x version constant ( )
Merge pull request from jpountz / fix / 12329
prepare release elasticsearch - cloud - aws - 1 . 0 . 0
Merge pull request from rmuir / harden_integration_tests
[ DOCS ] Add impact sections to the Go API docs ( )
Merge remote - tracking branch', "'", 'origin / master', "'
Remove DiscoveryNode # simpleMatchToFullName and method . ( )
[ TEST ] Harden AwaitsFix in SearchFieldsTests . testUidBasedScriptFields .
[ TEST ] Fix RepositoryURLClientYamlTestSuiteIT
Don', "'", 't register SearchTransportService map ( )
Merge pull request from jpountz / ingest_plugin_api
Rename bin / plugin from bin / groovy
[ DOCS ] Merges Metricbeat pages for breaking changes ( )
[ CORE ] Add unique lifecycle parser to EngineHolder
expose shards in the cluster state action
Added unit for
[ ML ] Add random log in the maintenance logging ( elastic / x - pack - elasticsearch )
Don', "'", 't double requests on RED shard nodes
Merge remote - tracking branch', "'", 'origin / master', "'
Merge pull request from cbuescher / introduce - shapebuilders
test : removed more merging in log log ( )
[ ML ] Use volatile boolean to stop blackhole process
Added support for acknowledgements to the primary shard id
Removed unused code
Alerting : Split alerting explain requests
[ DOCS ] Correct doc enrich API docs ( )
Shutdown : Add support for Ctrl - Close event on Windows platforms to gracefully .
Rename actions
Merge pull request from rmuir / nukeSettings
Mute AsyncSearchActionTests ( )
Merge pull request from talevy / more_details_factory_factory
Add fromXContent ( )
Query refactoring : query_string
[ TEST ] fix NPE in LdapTestCase
move field data
Change ShardFollowTask to RestExecuteWatchActionTests
Added the ability to the type in stats .
Fix typos in SequenceNumbersService . java
[ DOCS ] Clarify security security security ( elastic / x - pack - elasticsearch )
[ DOCS ] Add index . search API ( )
Remove unused User class from protocol ( )
Remove ALL unused CreateIndexRequest ( )
Add cloud - gce module
Merge pull request from elastic / revert - 18211 - docs / add_autosense_to_query_dsl
Stop ` full_id ` parameter early ( )
Mute', tests '( )
clean break of facets terms , closes .
[ DOCS ] Changes experimental flag to beta in datafeeds ( )
[ DOCS ] Reformat ` keyword_marker ` token filter ( )
Merge pull request from jimferenczi / copy_to_dynamic_object
fix asciidoc link
Remove old 0 . 90 shard allocator
[ DOCS ] Move migration APIs to indices . indices ( elastic / x - pack - elasticsearch )
Change ShardFollowTask defaults ( )
Change index . local . Custom to be set
Merge pull request from cbuescher / remove - matchall - norms
Fix LoggingOutputStream for Reconfigurator ( )
move to fix the cluster routing of the static manner
Apply deleting version check for StopDataFrameAnalyticsAction . Request ( )
optimize IndicesAdminClient # existsAliases to IndicesAdminClient # aliasesExist .
[ DOCS ] Add / _search_shards documentation
fixed compile errors due to upstream changes in elasticsearch itself
Test : Add unique logging to the tests
[ DOCS ] Call ILM index files ( )
[ ML ] Increase timeout to start stats management
[ TEST ] Fix compilation
Update the Introduced version to the tranform file
add missing Plugin to precommit
minor cleanups in SyncedFlushService
Merge pull request from jasontedor / remove - matchall - norms
Mappings : Rename the new field data method in the ` null_value ` .
with the tiny stemmer buffer when running it to read the node ( elastic / elasticsearch )
[ Transform ] Muted CcrRetentionLeaseIT - size . ( )
Merge remote - tracking branch', "'", 'es / master', "'", 'into enrich
Merge pull request from jimferenczi / copy_to_dynamic_object
properly support for write event api
body attribute was not required for flush
Remove dead code
SQL : Extend NULLIF ( expr1 ) method ( )
Recovery : finish the ability to control the number of shards in the gateway API
Tests : Make " constants " for watcher smoke tests
TESTS : Disable Dead Code in testRebuildLocalCheckpointTracker ( )
Updates to implement Writeable rather than Streamable
[ DOCS ] Warn about searching across a multi - level query ( )
Use ValuesSourceRegistry from ShardStateAction ( ) ( )
Fixed readme
Merge remote - tracking branch', "'", 'upstream / master', "'
Add Infrastructure to Javadoc . md in favor of existing JDK ( )
Merge pull request from talevy / more_details_factory_factory
don', "'", 't check if setting are present to the master , closes .
Watcher : Hide SchedulerEngine watch status ( )
Scripting : Fix IndexRecoveryCollectorTests
Merge pull request from rjernst / save_eclipse
Don', "'", 't reduce ` missing ` parameter in ` sum ` aggregation ( )
XContentBuilder from BroadcastOperationRequest
Cleanup nodes in SequenceNumbersService . java
first commit
Add Enrich imports
Add Javascript examples
[ DOCS ] Adds impact - plugin . es ( )
Update to elasticsearch 1 . 0 . 0
[ ML ] Increase timeout waiting for DFA jobs to finish ( )
[ CCR ] Remove Dedicated IndicesStatsCollector ( )
Fix version compatability test
Merge pull request from s1monw / fold_in_index_cache_module
Fix typo in path specification for / _cat / fielddata
Fix a Hashing
Avoid circular reference in exception
[ DOCS ] Clarify title for system indices ( elastic / x - pack - elasticsearch )
[ DOCS ] Update readme to use two docs ( )
When searching against an search of match_all filter
remove dead code
Docs : Reorganized migration links to the core topics ( )
Added removed imports
[ TEST ] Add additional logging to testCorruptTranslogFiles
Analysis : Add clear constraints .
[ TEST ] Fixed timeout for wait_for_active_shards test to allow
Added support for the rest test to not false .
Simplify NestedFieldComparators for numerics
Merge pull request from rmuir / spi_reload
Fix CoordinatorTests ( )
Don', "'", 't use keyword', "'", 's mapping reader
ILM : Add a simple function ( )
[ BUILD ] Ban all useage of Future # cancel ( true )
Recycler : cleanup bulk api should be able to transform the explain API .
[ DOCS ] Clarify Shadow replica setting
Remove Painless AST Lambda generation ( )
[ TEST ] Fix test bug to expectedTimeToHeal
HLRC : Add missing Lifecycle time ( )
Add integration tests
Merge pull request from rjernst / ingest_plugin_api
or some fixed support tests
Fix CorruptedBlobStoreRepository Test test
Raised IOException when profiling GetResponse . ( )
Merge remote - tracking branch', "'", 'upstream / master', "'
[ DOCS ] Fixes data frame analytics job ( )
Test : Add a simple test for the DeleteAlertResponse
Tests : Increase logging to flaky tests
Upgrade to lucene - 7 . 7 . 0 - snapshot - b313618cc1d ( )
Test : refresh only after the same test
Merge pull request from elastic / fix / indices - get - restspec
Merge pull request from rjernst / fix / 10659
Include autoscaling analyzers in release notes
Add feature of code ( )
Implement guice as read APIs
Use Date timeout in simple connection ( )
Merge pull request from rmuir / painless_generated_skip_checkstyle
[ DOCS ] Fix typo in get - related . asciidoc
[ ML ] Fix character node names in unencodable ( )
[ DOCS ] Adds link to bucket_span role mapping docs ( )
[ DOCS ] Fixes typo in CSV processor docs ( )
Add basic flag for experimenting with test ( )
[ DOCS ] Reformat ` keyword_marker ` token filter ( )
[ ML ] Remove support for deprecated field names when creating DF analytics ( )
Fixed sample code for minhash ( )
HLRC : Add Delete API to Calendar ( )
Add unit tests
Typo of', "'", ')
Merge pull request from rmuir / spi_reload
Allowing users to 6 . 2 ( )
Merge pull request from kkirsche / patch - 2
Fix testAutoExpandIndicesDuringRollingUpgrade ( )
Update resiliency . asciidoc
check not set soft - deletes on the test ( )
copy Actions to protocol ( )
when providing index creation , closes .
[ DOCS ] Fix typo', ( ') ( )
Merge pull request from jasontedor / shard - size
Bulk API : Move routing request
[ Internal ] make sure there are no duplicate index
Merge remote - tracking branch', "'", 'es / master', "'", 'into feature / ingest
Remove garbage
apply review comments from @ nik9000
Don', "'", 't try to some some case in the docs ( )
Docs : Add note about certificate management
More review comments
[ DOCS ] Warn about examples is now ( )
properly handle the script response , don', "'", 't break its on language caches
Multi data stream setting
Remove dead code
Merge pull request from talevy / more_details_factory_factory
Merge pull request from s1monw / cleanup_search_request
[ DOCS ] Mention Integer status is not in 1 . 2 . 0
[ DOS ] Fix typo in CSV processor docs ( )
[ DOCS ] Fix typo in RoutingNode migration ( )
[ DOCS ] Updated ` node . asciidoc ` index ( )
Merge pull request from rmuir / harden_integration_tests
Remove unused imports
Merge pull request from rjernst / standalone_tests_intellij
removed javax . mail from test dependencies
[ CCR ] allow auto - deletes ( )
Remove unnecessary writer Response ( )
Add integ - test dependency
[ TEST ] fix apache ds cleanup issues
Remove PreProcessModule
Update to elasticsearch 1 . 3 . 0
Fix docs style .
Conditionally repository - only fixture as 39525 ( )
[ ML ] Skip MlMetadata in DataCountsReporter ( elastic / x - pack - elasticsearch )
Corrected test bug
Merge pull request from ywelsch / fix / compressedstring_equals
[ TEST ] Allow to more a better failure
[ TEST ] temporarily log the default locale in REST tests
Sort simplifications to review ( )
Enable assertions to work on upgrade ( )
Remove deprecated token filter .
Merge pull request from nik9000 / max_merge_settings
[ DOCS ] Fix typos in synonym API docs ( )
[ Transform ] Update job stats tests to separate ( )
Recovery : allow maxUnsafeAutoIdTimestamp and Whitelists ( )
clean boost in query , closes .
reindex : fix index mapping mapping ( )
Update to AWS SDK 1 . 7 . 13
Enable license manager .
Ignore eclipse build dependency
[ Transform ] disable BWC as preparation for ( )
Build : Remove extra plugins link
[ DOCS ] Update term APIs ( )
Remove dead code
[ DOCS ] Fix typo in get role mappings API ( )
Upgrade to Lucene 4 . 6
[ DOCS ] Add missing icons to fix 8 . 0 breaking changes ( )
Merge pull request from jpountz / upgrade / lucene - 5 . 2 . 0 - snapshot - 1673124
Merge pull request from ywelsch / fix / index - api - restore
Add build short hash to the download manager headers to identify staging builds
Increase ensureGreen when index is active ( )
Merge pull request from jasontedor / remove - CONF_FILE
Remove )
Fix nested mapper on unmapped type ( )
Tests : Remove unused test that uses groovy tests
Fix Incorrect Scope ID in windows
Watcher : Remove catalog conversion from InternalClient
Fix RollupJobTaskTests handing of Slowness ( )
Remove @ AwaitsFix ( )
[ TEST ] fixed SimpleQueryTests # testDateRangeInQueryString to control the mappings upfront and wait for green
Upgrade to 6 . 0 . 0 - snapshot - 5c4168d ( )
Fix exact - get log for HLRC ( )
[ TEST ] Fixed suggestion tests in Ruby . upgrade
Merge pull request elastic / elasticsearch from rjernst / search_parser
Fix BWC issue a typed_keys
Add cliSetup calc to get test memory ( )
Remove NoneGateway from cluster state .
wrap test
[ TEST ] Removes version of / _nodes / missing
Build : Fix build - 27 ( )
Mute FullClusterRestartSettingsUpgradeIT
Lucene the TESTING specification around the cluster iteration
[ TEST ] Add additional assert to testCorruptTranslogFiles
fix calendar support for nested and use correct types ( )
[ Test ] Implement SuppressFrobidden annotation ( elastic / elasticsearch )
[ DOCS ] Fix typos in normalizers ( )
[ TEST ] don', "'", 't expect acks when isolating nodes
Add support for auto_generate_synonyms_phrase_query in query_string queries ( )
Remove', ['HLRC API from LifecyclePolicy ( )
[ ML ] Mute data frame analytics BWC tests ( )
Fix build
Merge pull request from s1monw / cleanup_search_request
Consolidate query generation from QueryShardContext
Merge pull request elastic / elasticsearch from rjernst / rest_alone
Aggregations : Adds MonitoringClientProxy
Query refactoring : NotQueryBuilder and Parser
Merge pull request from jpountz / upgrade / lucene - 5 . 4 . 1
Enhance tests for groovy commit .
Merge remote - tracking branch', "'", 'origin / master', "'
[ TEST ] add missing tests for search . allow_expensive_queries .
Merge pull request elastic / elasticsearch from colings86 / refactor / aggRefactoringChanges
Initial improvement this - databind tests
CCR : Requires soft - deletes on IP files ( )
[ DOCS ] Add cat', - 'level . asciidoc
Add versions 5 . 0 . 3 after release
Update to forbiddenapis changes .
Merge pull request from rjernst / ingest_plugin_api
Updated elasticsearch - to - 1 . 5 . 1 SNAPSHOT
Fix Painless conflicts
Allow retention lease actions ( )
[ DOCS ] Adds performance considerations section to transforms overview ( )
[ DOCS ] Add impact sections to snapshot / restore API ( )
make sure to close the default analyzer .
remove unused connector
Remove left index . type . ( )
[ DOCS ] Fix typo in RoutingNode comment .
[ ML ] Move model actions to use a same ( elastic / x - pack - elasticsearch )
SQL : Change javax and NamedExpression ( )
Merge pull request from jasontedor / search - sort - redux
add plugin to . gitignore
Fix bug in the composite config test ( )
keep indices and update_by_query
Don', "'", 't try to use system jna for naming conventions
Merge pull request from llasram / hll - estimate - bias - nn - k - 6
Merge pull request from rjernst / feature / bool_fields
[ TEST ] add additional logging to testCorruptTranslogFiles
Fix CloseIndexIT . testRelocatedClosedIndexIssue ( )
Security : document no index fields for runtime fields ( )
Invalidate mapper of date field
Remove [ D ] on - snapshots and Set ( )
[ ML ] Ensure total hits are tracked ( )
[ TEST ] Move to close the intended constructor ( elastic / x - pack - elasticsearch )
add lang - east - 2 . 0 - SNAPSHOT
Docs : Documented cross document settings in cluster state API ( )
[ DOCS ] Fix typo in getting started page
[ TEST ] Fix test for use comp tests
Merge pull request elastic / elasticsearch from dliappis / master
[ DOCS ] updated query took API ( )
Dependencies : Update javax . mail in watcher to 1 . 6 . 2 ( elastic / x - pack - elasticsearch )
moving version to 1 . 0 . 0 - Beta1
Disable BWC tests while merging
minor cleanups in SyncedFlushService
Merge pull request from rjernst / standalone_tests_intellij
Don', "'", 't let IllegalInstantException to determine DST gap
SQL : MATCH to new RestClient
SQL : Fix column size for SYS COLUMNS : ( elastic / x - pack - elasticsearch )
Convert', Eclipse 'mappers to new MapperTestCase ( )
[ TEST ] Add a test for empty proper fields .
Added the API definition for ` _cache / clear `
[ DOCS ] Add PIT to search type docs ( )
Fix CoordinatorTests ( )
Merge pull request from ywelsch / fix / handling_index_state
Security : don', "'", 't call prepare index for reads ( )
Remove basic feature checks from license state ( )
disallow deleting lifecycle policies for eclipse ( )
Removes the rest client for rest client name
Configuration : Add support for Ctrl - Close event on Windows platforms to gracefully shutdown node
Remove 5x from type ( )
Remove fixture interface from GCS Repo ( )
Enable customizing history cluster settings ( )
[ DOCS ] Add abbreviated titles
Merge pull request from rmuir / harden_integration_tests
Merge pull request from rmuir / more_indy_typing
Fully retention lease operations after cancelled ( )
security : mention comma - separated for IP and DNS name prompts
[ TEST ] Add SAML Redirect Signing test ( )
Add brief plugin to docs
[ DOCS ] Fixed typo ( ) ( )
remove stale nocommit
Moved the filter
sync , closes .
Move terms score to use ParseFields
[ TEST ] fix CommonTermsQueryBuilderTest
define transport source documentation for 2 . 0
[ DOCS ] Add xref to index suggest
Alerting : Split alerting forecasts into packages
Change BlobPath . buildAsString ( ) method
[ DOCS ] Correct get get settings docs for data stream docs ( )
Add docs for percolate query types ( )
[ DOCS ] Rewrite ` kstem ` token filter docs ( )
[ DOCS ] Fix typos
latest jsr166 upgrade only compiled with 1 . 7
[ DOCS ] Fixed note and X - Pack Kibana notes
Fix a bind bootstrap check ( )
[ DOCS ] Add watcher configuration to Elasticsearch packages ( )
Remove ClientHelper ( )
Cluster - circuit rebalancing of indices / documents in index requests ( )
clean term query , closes .
[ TEST ] Added missing test to testCorruptTranslogFiles
[ ML ] Ensure data frame analytics jobs don', "'", 't be', used '( )
fix build
[ DOCS ] Add docs for " mini " ( )
[ DOCS ] Fix typo in get data frame docs ( )
Fix testAutoFollowing timeout test ( )
Delete bootstrap check on Windows ( )
Remove explicit bwc test explicit ( )
Merge pull request from jdconrad / init2
Added boost_mode to rest interface
HLRC : Add not support for field field ( )
Add integration tests for the follower response ( )
[ TEST ] Don', "'", 't rely on ` new Date ( elastic / x - pack - elasticsearch )
create a default a type to the type script as well , closes .
Merge pull request from rjernst / save_eclipse
[ DOCS ] Add link to the Go snapshot page ( )
Make remaining tests more robust .
upgrade to jackson 1 . 7 . 5
[ TEST ] Ensure that IndicesRequestTests is repeateable ( elastic / x - pack - elasticsearch )
Merge pull request from rmuir / more_indy_typing
Fix ( )
Merge pull request from s1monw / renew_syn_on_merge
[ Zen2 ] Use no licensing for groovy plugin manager
[ DOCS ] updated cat remote tasks ( )
Merge pull request from rjernst / standalone_tests_intellij
bump to directly run for master
[ DOCS ] Add xref to multiplexer token filter docs ( )
Fail last comment
[ TEST ] Add index . type REST tests to a node level
cache md5 API , closes .
Don', "'", 't update SearchTransportService handlers when updating of entries
Build : Make idea Gradle ports
Remove', " 'plugin', " 'cluster action ( )
Merge pull request from ywelsch / fix / transport_index_duplication
Fix testSeqNoCollision ( )
Small typo in the docs ( )
Fail dynamic shards when ack times out
Bulk processor # awaitClose to close scheduler ( )
[ DOCS ] Enabled link in X - Pack install
Remove FileSystemUtils # maxOpenFiles
SQL : Fix typo in SQL docs
Query refactoring : query_string
Aggregations / HL Rest client fix : missing scores ( )
Merge pull request from s1monw / fold_in_index_cache_module
Add settings updater in the OffsetTime log ( )
allow to control the index level timeout in the node node
Lucene : Upgrade to latest 3 . 0 . 2 version , closes .
Fixes testThatNonExistingTemplatesAreAddedImmediately ( )
painless : Add a simple id to whitelist .
Move reader into backport ( )
fix typo
Merge pull request from elastic / upgrade / stricter_metadata_parsing
move to Lucene 4 . 9 . 0 . Beta1
Merge pull request from jasontedor / immutable - collections - be - gone
[ DOCS ] Add redirect for heap service . ( )
[ DOCS ] Change data frame analytics APIs ( )
Fix TransportSnapshotsStatusAction comments
Adjust Azure integration tests in Windows ( )
Query DSL : Allow to default certain type in query_string / field queries , closes .
[ ML ] [ Inference ] Unify top_classes object field names ( )
[ DOCS ] Add data streams to index APIs ( )
AwaitsFix IndexShardTests # testDocStats
[ TEST ] Revert accidential massive iterations
[ CCR ] Remove unused method ( )
[ DOCS ] Reformat cat', "'", 's json ( )
Add get 5 . 5 . 3 version
include_in_all default a index is backed in PlainOperationRounting
apply feedback from @ jaymode
Docs : Update version . asciidoc
[ ML ] Correctly fetch persistent tasks ( elastic / x - pack - elasticsearch )
Fix elasticsearch - cli dependency
Return DiscoveryModule instead of ClusterService to obtain local handlers .
[ DOCS ] Reformat script query to use unused query API ( )
Depend on elasticsearch - parent
doc
Update skip after backport ( )
Merge pull request from rjernst / buildsrc_test_cleanup
Upgrade to amazon for jsr166y changes ( )
[ TEST ] add a lot of forgotten forgotten added operations
SLM underscore HLRC Test ( )
Replace the TreeMap in the tests
[ ML ] Remove intervening interface ( elastic / x - pack - elasticsearch )
Consolidate cut into integrating more CancellableThreads ( )
[ Transform ] disable BWC as preparation for ( )
Adjust soft - deletes version after backport into 6 . 5
[ DOCS ] Add - index sentence
Revert " [ TEST ] Enable skip compatability tests for search request ) ( elastic / x - pack - elasticsearch )
[ DOCS ] Merges duplicate pages for Active Directory realms ( )
Add test for Log4j throwable proxy leniency
Improve Expected', ( 'defaults ) method to isDataNode
[ TEST ] Add additional logging to testCorruptTranslogFiles
lucene4 : only use of byte settings
Merge pull request from mrsolo / features / thirdparty
Removed dead code
Support doing _routing of the node when allocating the shard , closes .
[ TEST ] Ban brackets
[ DOCS ] Reformats cluster allocation API API ( )
[ TEST ] fix CommonTermsQueryBuilderTest
Tests : Fix test parsing for timevalue first
update skip after backport of ( )
Add fromXContent method to SuggestBuilder
Merge remote - tracking branch', "'", 'elastic / master', "'", 'into feature / sql
Enable license build
Merge remote - tracking branch', "'", 'origin / master', "'
[ STORE ] Calculate Alder32 Checksums for legacy files in Store # checkIntegrity
Merge pull request from rjernst / deprecate_exception_hurter
Unzip quietly and Tokenizers ( )
Opt : Adapt ( )
SQL : Add IsNull support to ` sql ` Test .
Enhance field stats to use correct
add some ensureGreen to aggs tests
Remove TimeSeriesLifecycleActionsIT . setType ( ) ( )
Merge pull request from rmuir / eight_point_three
Merge pull request from javanna / enhancement / simulate_index_type_not_required
[ DOCS ] Fixed typo ( ) ( )
Fix unchecked compile errors in ObjectParser
[ TEST ] Fix test bug in PluginCliTests on Windows
Merge pull request from llasram / hll - estimate - bias - nn - k - 6

test : fix rolling test for the use of the new segment
add an index to delete - allocation actions , closes .
[ DOCS ] Fix cannot - typo in the highlighter docs .
Fix compilation error due to upstream change
[ TEST ] Only wait for fields to have been applied in the mapping .
doc
[ CORE ] Don', "'", 't set the setting by default
[ BUILD ] Randomize - > from 14420d7c4e15df9b565b50ef5beab797f756c3ac .
[ client ] Remove Dedicated code ( )
Upgrade plugin to 10 0 . 4 . 1
Fix typos in Groovy graph API docs
Enable BWC version check for the rolling snapshot ( elastic / x - pack - elasticsearch )
Build : Remove plugin cyclic dependency ( )
[ DOCS ] Creates delete default overview ( )
Watcher : Properly when watch is configured ( )
Non - based environment
Watcher : Allow to new an process with type ( elastic / elasticsearch )
[ STORE ] Calculate Alder32 Checksums for legacy files in Store # checkIntegrity
[ ML ] Remove partition_field', gitigno
lucene 4 : Upgraded ContentIndexSearcher
Remove binary license
minor cleanup processing
Remove heuristics for intervals highlighter ( )
Merge pull request from rjernst / remove / smart - object - wrapper
[ DOCS ] Update docs for a plugin plugin ( )
Scripting : Add trap input ( )
Register new query for terms query ( )
SQL : fall handling for null functions ( )
Fix compilation issue in node . logging ( elastic / x - pack - elasticsearch )
[ Test ] ElasticsearchIntegrationTest . clearDisruptionScheme should test if the current cluster is internal
add a test', for 'terms facet
Adjust bwc version after backport of ( )
Merge pull request elastic / elasticsearch from jasontedor / sets - be - gone
Mapper : An analyzer mapper allowing to control the index analyzer of a document based , closes .
Merge pull request from ywelsch / fix / index - delete - restspec
Fix bwc tests after backport of ( )
Remove unused constants
[ DOCS ] Reformat index template API docs ( )
[ DOCS ] Update migration docs to migration docs ( )
Fix BytesStreamInput .
Fix casing of " Elasticsearch " in how - to docs
Fix testThatNonExistingTemplatesAreAddedImmediately ( )
Merge pull request from jimferenczi / upgrade_lucene_1714615
fix request handling in completion suggest tests ( )
[ DOCS ] Reformat ` keyword_marker ` token filter ( )
Disable specific file specific settings , closes .
Remove SearchRequestParsers ( )
Merge pull request from cbuescher / feature / query - refactoring - constantscore
Fix CoordinatorTests helper in test
Fix ParseFieldMatcher usages in SearchableSnapshotAllocator ( )
[ ML ] Remove encapsulated bwc serialization on ML request ( elastic / x - pack - elasticsearch )
Bulk processor # awaitClose to close scheduler ( )
Add HLRC support for enrich execute policy API ( )
add support for STANDARD license
[ DOCS ] Fixes shared attribute for simple token breaking changes
docs : Remove the ability to use a document name .
Merge pull request elastic / elasticsearch from rjernst / deguice1
[ DOCS ] Fixed typo in jump link . ( )
Add CoordinatorTests ( )
Merge pull request from rjernst / buildsrc_test_cleanup
no need to check for empty types in the file name
Merge pull request from ywelsch / fix / handling_index_state
Remove unused printStackTrace
Removed loopback comment
Merge pull request from rmuir / harden_integration_tests
[ TEST ] add SuppressSysoutChecks to DistributorDirectoryTest
prepare release elasticsearch - cloud - aws - 2 . 0 . 0 . RC1
Merge pull request from rmuir / rat
Watcher : Make SchedulerEngine job handling threadsafe ( elastic / x - pack - elasticsearch )
Remove return for the Painless method . ( )
add version 6 . 2 . 3 with bwc indices
Update search . percolate . disable_purge to be set
Merge remote - tracking branch', "'", 'upstream / master', "'
Fix compilation errors due to upstream changes ( elastic / elasticsearch )
Merge pull request from rjernst / fix / field - query
Remove unused imports
[ TEST ] Fix test bug to expectedTimeToHeal
Tests : suppress testIndexUpgradeInfoLicense on release builds ( elastic / x - pack - elasticsearch )
Add Gradle ( )
make sure we close the new cluster state version
optimize single http checks
Build : Add fixture capabilities to integ tests
removed javax . mail from test ( )
[ DOCS ] Tidied up anchors in restore . asciidoc ( )
 )
Mute failing tests in FIPS mode ( )
Document max license
Remove unused method
Fix test to use windows ( )
[ ML ] Skip calendars with job groups in find_file_structure ( elastic / x - pack - elasticsearch )
Merge pull request from jasontedor / immutable - collections - be - gone
add missing toString to FsInfo # factor
[ Transform ] simplify leader lifecycle monitoring ( )
Remove MappedFieldType . setType ( ) ( )
[ DOCS ] Add PIT section to the Go docs ( )
Merge pull request from ywelsch / fix / index - settings - restore
[ DOCS ] Added a few " score_mode " to the docs .
Remove PreProcessModule
[ script ] Update to elasticsearch 1 . 0 . 0
Fixed sample typo
added removed trace logging
Remove PreProcessModule
security : cleanup compilation in actions , closes .
Tests : Muted IndexStatsIT
Merge pull request from s1monw / serialize_interrupted_exceptions
Remove unused constant

[ ML ] [ Inference ] changing cancel failure ( )
Fixed the whitespace in the JSON api rest tests .
Merge pull request from cbuescher / feature / add - get - phases
Queries : Optimize failure when a type changes it , closes .
Handle painless scripts when script is not available
Rename packaging tests to use testclusters ( )
Merge pull request from s1monw / handle_source_canceled
Internal : adjusted visibility to package private for BroadcastShardOperationRequest subclasses and their constructors
add more debug logging for testRunningTasksCount
Fix snapshot flag accidentally switched for 1 . 0 beta2
Mute failing test
[ ML ] Increase test for starting jobs tests ( )
Aggregations : Add cumulative sum aggregation
[ CORE ] Rename var to use the block message
Remove GceMetadataService interface as not needed
d 'StackOverflowException for connect .
Merge remote - tracking branch', "'", 'es / master', "'", 'into feature / ingest
Add Enrich Origin ( )
Sorting on _score in the cluster state , closes .
Fix checkstyle in test
Clean up tests
Painless : Remove hack utility - by - ClassCastException .
Update to Lucene 4 . 6 . 1 / Elasticsearch 1 . 0 . 0
Fix TransportSnapshotsStatusAction Loops in ExceptionsHelper # unwrap ( )
Add missing words in docs ( )
Add extra Retry support for experimenting on test ( )
Upgrade to Netty 4 . 1 . 25 ( )
Fix strict logging for deprecations in shield
Exclude BWC check in peer recovery ( )
[ ML ] Remove unused Request constructors ( )
Remove garbage
Fix synonym terms aggregation .
Added index / index . settings . allocation and fs settings that 5 . 2
Update', with 'a separate API page from 7 . 0
Remove Redundant clusterService in SnapshotShardsService ( )
Added API for multi - cluster search request
Add version
[ DOCS ] Revise machine learning indices to the value definition ( )
Make DeleteStep retryable ( )
[ DOCS ] Add ML configuration section
Cut
Make DeleteStep retryable ( )
Cherry pick tests and seqNo recovery hardning from
Fix ( )
[ DOCS ] Fix typos
improve simplifications in test builder tests
Add backwards compatibility indices
Enable Build version for Windows ( )
Raised IOException on deleteBlob ( )
nicer logging
Add built - in user and role for code plugin ( )
Merge pull request elastic / elasticsearch from dliappis / master
Tests : Fix IndexRecoveryCollectorTests
nicer logging
Merge pull request from javanna / enhancement / migrate_node_settings
Merge pull request from jdconrad / stype
Merge pull request from rmuir / nukeSettings
[ TEST ] Use SecurityClusterClientYamlTestCase ( elastic / x - pack - elasticsearch )
[ TEST ] fix testBulkRequestModifier to AbstractStreamableXContentTestCase ( )
[ TEST ] add additional logging to testCorruptTranslogFiles
Make ` RoutingAllocation ` for ` _create ` . ( )
Merge pull request from llasram / hll - estimate - bias - nn - k - 6
[ DOCS ] Added link to docker API
Fixed sample typos to query and docs
[ DOCS ] Add data streams to Elasticsearch Reference ( )
security : cleanup IndexPrivilegeTests after removal of predefined privileges
Enable TestClustersPluginIT . testMultiNode ( )
nuked ESLicenseProvider
Added support for acknowledgements to the local API case
Mute TimeSeriesLifecycleActionsIT . testRolloverAlreadyExists ( )
refactor nowInMillis to search error
Recycle buffers ( )
Setup delete API to move the index
[ TEST ] Add additional logging to testCorruptTranslogFiles
Remove ClientHelper ( ) ( )
Merge pull request from rmuir / nukeSettings
Fix packaging
Shard Allocation : Allow to control how many cluster wide concurrent rebalance ( relocation ) are allowed , default to 3 , closes .
Merge pull request from nik9000 / grep_minus_e_portability_fix
Merge pull request from s1monw / renew_syn_on_merge
Fixed licence issue to docs creation docs
[ DOCS ] Add xref to multiplexer token filter docs ( )
Clear single cluster and flush local shard once we be added , closes .
Multi - data not index fails , closes .
ingest : Update KV joda . py
Silence CcrRetentionLeaseIT # testForgetFollower
Merge remote - tracking branch', "'", 'elastic / master', "'", 'into zen2
Fix in reload test test ( )
[ DOCS ] Merges Metricbeat system module ( )
Replacing that prevent the exception that has been used .
[ DOCS ] Fixed typos
[ TEST ] Add REST API test to HLRC ( )
[ DOCS ] Merges duplicate link to PKI realms ( )
Type is a specific node ( )
HLRC : ML Add Job to Calendar API ( )
add comparator to ensure we can exit
Core : don', "'", 't include a more warmer aggregator .
Merge pull request from nik9000 / java_8_for_package_tests
Upgrade 8 . 7 . 0 - snapshot - 5c4168d ( )
[ TEST ] Add a dedicated test for empty indices in InternalEngineIntegrationTest
Merge pull request from robertberg / rberg . searchdocs
Merge remote - tracking branch', "'", 'origin / master', "'
[ DOCS ] Fixed docs for search template API docs ( )
doc
[ TEST ] fix CommonTermsQueryBuilderTest
[ discovery ] Check when field is in same settings
Add ability to associate version ( elastic / x - pack - elasticsearch )
[ DOCS ] EQL : Document ? wildcard for Asciidoctor ( )
[ DOCS ] Fix X - Pack tag checks ( )
Fallback to implement Writeable rather than Streamable
Minor simplifications in SyncedFlushService
Remove references to old version ( )
Consolidate already a cleanup to commit when we can ignore skipTranslogRecovery in IndexShard
ResourceWatcher : Add support for terms field to the text aggregation
Added a debug logging to help caches
Throw a number of field message
fix compile errors due to upstream changes .
Fix P . testForgetFollower
Updated java to elasticsearch 1 . 0 . 0
[ Transform ] disable BWC as preparation for ( )
Add progress for enrich autoscaling policy tests
[ TEST ] fix CommonTermsQueryBuilderTest
Remove registerQueryParser
Merge pull request from biolds / master
Add docs for " authorization_realms " ( )
[ DOCS ] Add missing comma
Remove explicit comment
Aggregations : Remove ` ordinals ` option .
Add tests for es . 19 . 0 . alpha1
Reject pattern name in range pipeline ( )
ip : Forgot to prevent the AlertsClient tests
[ DOCS ] updated typo in search - aggregation . asciidoc
Fix incorrect removal of ScriptScoreFunction in FiltersFunctionScoreQuery
Add elasticsearch - - - module ( elastic / elasticsearch )
Add PERL # Resume ( ) ( )
[ TEST ] Reduce intensive event time
[ ML ] relaxing parameters for stratified job test ( )
Marvel : Add MonitoringClientProxy
Improve constants
[ TEST ] remove AwaitsFix in testSnapshotSucceedsAfterSnapshotFailure .
[ TEST ] mute SearchHitsTests to AbstractStreamableXContentTestCase ( )
HLRC : Add Put Lifecycle Policy API ( )
Merge pull request from biolds / master
Fix the NPE of ScriptScoreFunction in FiltersFunctionScoreQuery
Add async_search parameters to HLRC ( )
Text Query : don', "'", 't copy the language version
Added multi - type to o . e . action . Beta1
fix IntelliJ IDE options ( )
Merge pull request from s1monw / cleanup_search_request
Fix FullClusterRestartIT . testRolloverAlreadyExists ( )
Improve import
[ DOCS ] Add ML configuration details ( elastic / x - pack - elasticsearch )
Wire up IpRangeAggregation to ValuesSourceRegistry ( )
update to 0 . 12 . 0 SNAPSHOT version
Upgrade ASM for 8 . 0 ( )
[ DOCS ] Add xref to multiplexer processor ( )
only if set value is not supported
Fixed sample code for minhash ( )
use existing inputstream to read the settings
add ` beforeIndexAddedToCluster ` callback
upgrade to amazon aws jdk 1 . 7 . 2 ( )
Update to Lucene 4 . 10 . 2
[ DOCS ] Add - note to HLRC HLRC docs ( )
[ doc ] Remove support from mapping contexts ( )
[ DOCS ] Fine - tunes training_percent definition . ( )
Use - from wildcard for resolve actions
Add missing examples
Build : Update setup . asciidoc
 )
Queries : add a validate mapping to serialize the type cache on a field on each type , closes .
Add field to field_value_factor
support privileges in the tcp classes
[ Security ] Remove dead delete job index ( elastic / x - pack - elasticsearch )
SQL : Fix name of extracting data type ( )
Remove SearchPhase interface ( )
[ DOCS ] Reformat ` search . max_buckets ` ( )
move to 1 . 2 . 0 snap
[ TEST ] use SecurityClusterClientYamlTestCase ( )
S3 Repository : Remove Snapshot Code ( )
[ TEST ] Improved IndexAliasesTests
Remove unused modifier
Merge pull request from rjernst / save_eclipse
[ TESTS ] Mute IndicesCustomDataPathTests
more spelling
Introduce Gradle quietly
[ CCR ] Add create_follow_index privilege ( )
[ DOCS ] Fixes detection APIs ( )
[ DOCS ] Add links to Beats indices docs ( )
Improve import
Unmute SearchWithRandomExceptionsIT
Add Intellij support for testRunningTasksCount
Internal : Remove the ` node .', config '. allow_explicit_index ` setting
Merge pull request from spinscale / 1503 - cleanup - remove - bytesarray - unsafe
[ DOCS ] Reformats nodes for cluster stats API ( )
Gracefully pick tests for new remotes
Fix handling of UpdateIT
Internal : Ensure that explanation descriptions are not null on serialization .
Fix error message for MappingUpdatedAction
Minor changes in ScriptExceptions
Remove a settings_require_units ( . . . ) method
[ PARSER ] [ 1 . 0 ] version
[ CCR ] Include tests for follow follow requests ( )
[ ML ] [ Inference ] Add necessary ( )
update to 0 . 12 . 0 SNAPSHOT version
Remove old support for wildcard ( )
Fix PrimaryAllocationIT # testForceStaleReplicaToBePromotedToPrimary ( )
[ TEST ] Add ML test for QueryShardException
Add shuffling xContent to aggregation tests
[ TEST ] Check QueryProfileShardResult time
[ TEST ] Renamed Lucene 4 . 10 . mode . Beta1
Enforce that we check back to Lucene ( )
[ ML ] Use PersistentTasksService # waitForPersistentTaskStatus ( . . . ) instead of PersistentTask # needsReassignement ( . . . ) for checking whether a job task is stale when )
Remove unused User method from protocol ( )
[ TEST ] fix ClusterApplierService # testClusterStateUpdateLogging ( ) ( )
[ TEST ] Add SuppressSysoutChecks to DistributorDirectoryTest
[ TEST ] Switch to deprecated native license
Added ` top_hits ` query ( )
Use explicit index / templates
Transport : Add use of the alert manager to the correct dir
[ DOCS ] Fixes missing comma
Small text API ( )
[ TEST ] Fix too null for greaterThanOrEqual fix
add a examples
[ ML ] Mute data frame analytics BWC tests ( )
Add Assertions on GeoDistanceSearchBenchmark ( )
Remove', ['painless tests only as it is unzipped
Date : Add DateFormatters class that uses java . time ( )
Revert " Build : Allow plugins to " upgrade " provided deps "
[ TEST ] Add IndexShardOperationPermits to DistributorDirectoryTest
Merge pull request from gmarz / fix / handling_index_state
Reinstate link to the 5 . 0 . 0 version
Fix PrimaryAllocationIT # testForceStaleReplicaToBePromotedToPrimary ( )
HLRC : Remove Optional in HLRC API ( )
[ ML ] [ Inference ] adding starting * finder ( )
SQL : Make Literal for NamedExpression ( elastic / x - pack - elasticsearch )
Move tests to JUnit
Removed unused code
[ DOCS ] Add tab widget for boolean fields ( )
remove use of ImmutableList in shield
[ DOCS ] Clarify X - Pack installation ( elastic / x - pack - elasticsearch )
Removes Math . Request . type ( ) ( )
fix compilation error
Merge pull request from jdconrad / cast
Merge pull request elastic / elasticsearch from rjernst / rest_alone
Disable BWC for 2 . 0 . 0 - SNAPSHOT
Merge pull request from rmuir / rat
Add data stream to ShardFollowTaskReplicationTests ( )
[ ML ] Increase not fix when creating time setting ( )
Include the ml inference aggregation doc ( )
Add method ( )
Merge pull request from javanna / enhancement / bool_fields
[ BUILD ] Add licensing permissions to integration tests
[ ML ] Remove support for regression in data frame analytics ( )
[ DOCS ] Reformat ` query ` query ( )
review comments
Do not warm all test timeout in tribe tests ( )
Docs : Add Fedora - format . asciidoc
[ DOCS ] Add data streams to main docs ( )
Fixed docs for date math in reindex
Remove unnecessary settings file from nodes ( )
Merge pull request from jasontedor / immutable - split - version - 6
Fix daemonization shard state test
[ TEST ] Fix ClusterApplierService # testDateRangeInQueryString ( )
update to 0 . 19 . 0
Merge pull request from rjernst / save_eclipse
[ ML ] MlMemoryTracker should wait for job timeout in tests ( )
Fix ( )
Add support for all plugins in the settings validation ( )
[ DOCS ] Add missing widget for Asciidoctor migration ( )
Adjusted Exception in NetworkUtilsTests ( )
max bucket reducer tests
Alerting : Fixed IndexRecoveryCollectorTests
Internal : Remove injection of unannotated filtering in the static manner
even on failure when recovering to recover , closes .
Remove the docs setting for indices_boost . ( )
Remove ( )
Treat ActionListener and PendingClusterTasksResponse ( )
[ TEST ] Check accounting breaker is equal to segment stats rather than 0
Text Query has been replaced by Match Query
Upgrade to lucene - 7 . 0 . 0 - snapshot - a0aef2f
Fix test
doc
Upgrade to Lucene 4 . 8 . 0 - snapshot - 08b8d116f8f ( )
reenable bwc test now that is merged ( )
Account implementation for es . 0 . 0 . Beta1
[ DOCS ] Note that required
[ TEST ] Use assertAcked in random queries to simple test setup
[ http ] make sure that watcher is never available
[ Docs ] Add clarification', in 'examples of examples . ( )
Remove BufferedDeletesStream
Update to elasticsearch 1 . 0 . 0 . Beta1
[ DOCS ] Correct typos in X - Pack installation ( elastic / x - pack - elasticsearch )
Mute failing test
Make ValuesSourceRegistry immutable volatile .
[ TEST ] Add missing Redirect Signing often ( )
Settings : migrate discovery settings
Add the rest - spec for submodule . sh ( )
[ DOCS ] Add note about community clients ( )
add file support for the jar
[ CCR ] Add create_follow_index privilege to test ( )
Merge pull request from rmuir / nukeSettings
when ` index . auto_create_index ` setting , closes .
Fix typo in path specification for / _cat / fielddata
make the default type when merging
Docs : Augment javax . ( )
Fix test code test ( elastic / x - pack - elasticsearch )
fix compile check after release
[ DOCS ] Add index . max_buckets integration ( )
Merge pull request from ywelsch / fix / transport_index_duplication
Added multi - search - search warnings
[ Transform ] make sure does not return ( )
[ DOCS ] Note plugin . local . auto_import_dangled
[ DOCS ] Removes beta label in transforms docs ( )
[ DOCS ] Explain the remote search API docs ( )
Remove infinite Loop in SnapshotShardsService ( )
Remove PROTOTYPE from ExtendedBounds
Fixed sample code for minhash ( )
Aggregations : Adds cumulative constraints of field data
Remove references to ValuesSourceRegistry ( )
[ TEST ] Fix test bug in MultiOrdinalsTests .
Painless : Fix randomizedtest fail
[ TEST ] Enhance fields in SignificantTermsBWCTests
[ ML ] Update DataStreamDiagnostics classes to be memorySizeSettting ( )
Merge remote - tracking branch', "'", 'es / master', "'", 'into json_strict_duplicate_checks
Remove logging from ScriptService
Merge pull request from rjernst / buildsrc_ide
Prefer collectExistingBucket in MappingQueryParser
Checkstyle'] 'down search query refactoring
[ DOCS ] Adds the default flag
mute tests , see 37117
Remove dependency from forbidden - client - test ( )
SQL : Use NULLIF ( )
Tests : Fix systemkeytool failure in testRebuildLocalCheckpointTracker ( )
[ DOCS ] Fixes typo in jump settings
[ DOCS ] Add PGSync as community - supported integration ( ) ( )
Update bundled JDK version to 2 . 6 . 1 ( )
Update Java . asciidoc
[ DOCS ] Merges duplicate doc docs for script realms ( )
Add more debug info to experimenting on test ( )
Add versions 5 . 6 . 12 and 6 . 4 . 1
[ DOCS ] EQL : added a missing clarifications in EQL docs
Alerts : Fix use of nested templates
remove unused code
Only latest indices filtering
Fixed the new version in the classpath
SNAPSHOTS : Adjust BwC Versions in Restore Logic ( )
Fix SYS Math streams in SearchableSnapshotAllocator ( )
Bulk processor # awaitClose to close scheduler ( )
[ Security ] Remove Dedicated IndicesStatsCollector ( )
Reduce lowercase_expanded_terms in MappingQueryParser
Add integration tests
Move BWC version to 5 . 5 after backport
SQL : Make NamedWriteableRegistry immutable ( )
Alerting : Add ` publish_host ` aggregation to ` thread_per_shard ` .
[ DOCS ] Add redirect for ` search . max_buckets ` ( )
Fix PrimaryAllocationIT # testForceStaleReplicaToBePromotedToPrimary ( )
[ Tests ] Mute CcrRetentionLeaseIT . testRetentionLeaseIsRenewedDuringRecovery ( )
Move to Lucene 8 . 5 . 0 snapshot version
added missing getters to FsInfo . IoStats class
[ CORE ] [ 1 . 0 . 0 ] version
added the API definition for ` _cache / clear `
[ DOCS ] Reformat data stream limit API ( )
Test : wait for pending tasks to REST tests ( )
If that we skip out
Deprecate ` SearchIT # flatSettings ` ( )
fixed the test bug
test : don', "'", 't randomly wrap index reader
[ DOCS ] Clarify Metricbeat restore overview ( )
[ DOCS ] Add xref to multiplexer suggest
Merge pull request from s1monw / serialize_interrupted_exceptions
using the separate thread pool size for incoming indices ( )
make module immutable lenient .
externalize to control the index of the closed manner
[ DOCS ] Fix typo in get api docs ( )
Use different cluster settings for testOpenCloseApiWildcards
HLRC : Add support for data tiers API ( )
Merge pull request elastic / elasticsearch from rjernst / rest_alone
Add bwc support for pre - 5 . 0 . 0 - alpha4
Do not fail in multi - node ( )
Fix comments in AcknowledgeResponse
[ ML ] Mute data frame analytics BWC test after backport ( )
Fix integ tests for the right ( )
Merge pull request from rmuir / harden_integration_tests
fix a typo in _analyze docs ( )
Remove unused mapStringsOrdered
HTTP : Fixed IndexRecoveryCollectorTests
Replacing that ` search_analyzer ` is not consistent
Implement DataFrameAnalyticsAuditMessage # sizeOf ( ) ( )
[ DOCS ] Rewrite ` remove_duplicates ` token filter ( )
Merge pull request from rjernst / client_jar_id
[ TEST ] Add SuppressSysoutChecks to DistributorDirectoryTest
refactor 0 . 19 . 0
Remove references to deprecated type templates ( )
[ ML ] Change job to be memorySizeSettting ( )
add missing imports
When searching against an index / shards thread pool
[ DOCS ] Fix typo in CCR feature "
Bad asciidoc link
fix constants
Enable BWC and search shard failures ( )
addressed JvmErgonomicsTests for core
Merge pull request from ywelsch / fix / update - cs - with - routingresult
no not set mapping when it', "'", 's applied , closes .
Don', "'", 't fix the missing', IOException 'test . ( )
protected : mapping with a same of date type , closes .
update to 0 . 13 . 0 SNAPSHOT
Test : allow security actions to allow ( elastic / x - pack - elasticsearch )
Merge pull request from s1monw / renew_syn_on_merge
minor chagne
est : Add ability to put user in the password in the tests
Docs : Use task_id in the QUERY node to the guide
Fix unchecked compile errors and add scripts . ( )
[ DOCS ] Fixed links APIs ( elastic / x - pack - elasticsearch )
Add query response API
Update to elasticsearch 1 . 4 . 0
[ TEST ] packaging : use java code for invoking ( )
Fixed missing 8 . 0 . 3 version
Transport : Adapt IndicesClusterStateService to use new ids
[ DOCS ] SQL : Fix link to extract functions ( elastic / x - pack - elasticsearch )
Replace more fromXContent and Move ( )
Merge pull request from rjernst / fix / 12329
Merge pull request from s1monw / fold_in_index_cache_module
[ CCR ] Add simple with data frame analytics test ( )
[ DOCS ] Reformat unique APIs ( )
[ DOCS ] Add abbreviated titles
[ ML ] [ Data Frame ] allowing to specify that tasks ( )
Merge pull request from nik9000 / xlint_plugins
Update gradle to 6 . 1 . 1 ( )
Add unreleased v6 . 2 . 3 version
EQL : Add Delete License from Calendar ( )
Merge pull request from s1monw / cleanup_search_request
[ DOCS ] Linked cat - pending to cluster - pending
Docs : Fixed bad link in release notes
add getters realm
Docs : Fixed typo in query_string - cache to docs
Alerting : Split soft - deletes ( in milliseconds ) .
[ ML ] [ Data Frame ] adding model model policy ( )
lucene 4 : Converted ContentIndexSearcher
Do not be sync ( )
cancel recovery
Merge pull request from s1monw / move_refresh_into_index_service
[ ML ] Calendar event actions ( elastic / x - pack - elasticsearch )
Update to elasticsearch 1 . 0 . 0
Remove dead objects in IndicesStatsResponse ( )
Merge pull request from robertberg / rberg . searchdocs
Merge remote - tracking branch', "'", 'upstream / master', "'
Mapping type field , closes .
Remove vestiges of deprecated settings
Fix handling of version table when http to register http ( )
fix tests build ( )
Merge pull request from ywelsch / fix / compressedstring_equals
Fix merge ( )
move to 1 . 2 . 0 snap
S3 repositories : remove lucene test ( )
Add ability to active task ( )
Merge pull request from rjernst / module_culling3
Add back trace logging
rename indices sig
SQL : Fix subdirectories in JdbcResultSet ( elastic / x - pack - elasticsearch )
provide lastGc
[ ML ] Increase wait for templates timeout in tests ( )
Build : Fix license after icTests ( )
update documentation with release 2 . 4 . 1
[ TEST ] mute BWC test .
Remove redundant PutIndexTemplateRequestBuilder # localNode ( )
Merge pull request elastic / elasticsearch from rjernst / build_info
Docs : CONSOLEify plugin . asciidoc
Fix checkstyle error
Reenable Security test for 2 . 6 . 1 ( )
[ ML ] Remove redundant logging
Fix serialization for OSX
Document requests
[ TEST ] remove needless ClusterScope annotation from NettyTransportMultiPortTests
ogging : add logging to MockRepository on remote cluster states
Disable BWC tests while merging
[ DOCS ] Add abbreviated titles
Remove garbage
Remove leftover comment
Merge remote - tracking branch', "'", 'elastic / master', "'", 'into feature / sql
Merge pull request elastic / elasticsearch from PhaedrusTheGreek / group_search_noattrs
Update Java . asciidoc
Merge pull request from rjernst / unused
Upgrade to lucene - 8 . 7 . 0 - snapshot - 830bd186a8d . ( )
Rename local property to index name ( )
remove the unnecessary license build . sh ( elastic / elasticsearch )
Update matrix . asciidoc
[ Docs ] Fix typos in context aggregation docs
Merge pull request from nik9000 / grep_minus_e_portability_fix
Merge pull request from nik9000 / painless_generated_skip_checkstyle
[ DOCS ] Warn about add and 1 . 0 . 0 . 0
Docs : Update version . asciidoc
[ TEST ] remove support for indices . optimize
[ ML ] Rename action to trained model metadata ( )
SQL : Enable LEFT and RIGHT in DATEDIFF ( )
better shutdown issues
Rework REST tests for X - Pack scripts ( )
Merge pull request elastic / elasticsearch from rjernst / deguice1
Removed incorrect transport service
Add async_search get and delete APIs to HLRC ( )
Cut global , missing , cardinality , and value_count to registerAggregation
[ TEST ] Remove needless ClusterScope annotation from NettyTransportMultiPortTests
reverted simplified the ` type ` aggregator to the ` kstem ` aggregation .
[ DOCS ] Reformats cat indices API ( )
[ DOCS ] Add tab widget for examples . ( )
Remove SearchRequestParsers ( )
Tests : Add missing tests
[ DOCS ] Add xref to multiplexer fields for filters fields ( )
Update TESTING . asciidoc
Merge pull request from rmuir / shave_mustache
Remove DocumentFieldMappers # simpleMatchToFullName in favour of existing DiscoveryNodes # getLocalNode
[ ML ] Fix bug in MlStrings # hasValidLengthForId
SQL : Make Literal a NamedExpression ( )
prepare release elasticsearch - analysis - stempel - 1 . 8 . 0
[ DOCS ] Add missing widget for TLS / SSL settings ( )
Merge pull request elastic / elasticsearch from PhaedrusTheGreek / group_search_noattrs
Data stream search context API ( )
s / nocommit / norelease
[ CORE ] Calculate Alder32 Checksums for legacy files in Store # checkIntegrity
Docs : Remove docs for Painless / . . scripts ( )
Add UBI calc tests ( )
Merge pull request from abeyad / patch - 2
Add ImmutableMap $ import to its own dependencies
[ TEST ] Protect UpgradeIT from using too many replicas
[ TEST ] fix brackets
HLRC : Add String API ( )
upgrade to lucene 8 . 4 . 0 - snapshot - 08b8d116f8f ( )
Reindex : Do not use random index writer test
Merge pull request from rjernst / remove / smart - be - gone
Increase debug logging in the test
Made deleting rest / temporary threads in shield
Simplify testEnforceMaxFileDescriptorLimits for now
Add TODO for aggregating on _index .
Add switch to CCR docs for searchable snapshots ( )
Merge pull request from javanna / enhancement / 8964
Build : Fix randomizedtest fail
[ TEST ] Upgrade for AWS SDK to 1 . 1 . 0 - SNAPSHOT
add back trace logging
Cleanup comments in SequenceNumbersService . java
prepare release elasticsearch - cloud - aws - 1 . 14 . 0
[ TEST ] add a dedicated test for empty files
Remove SecurityTransportModule
[ DOCS ] elastic / monitoring - monitoring attribute
Tests : Fix TimeThrottleIntegrationTests to not rely on root actions ( elastic / x - pack - elasticsearch )
NullPointerException when creating an watch with Jira action ( )
[ ML ] [ Inference ] fixing ingest IT tests ( )
[ ML ] Minor the root_cause of error message ( elastic / x - pack - elasticsearch )
Remove signatures from the instantiable class ( )
No quartz thread pool size in node api
[ ML ] [ Inference ] Add ingest index privilege tests ( )
Test : don', "'", 't randomly wrap index reader
Upgrade to Netty 4 . 1 . 25 . Final ( )
Merge pull request from jasontedor / joiner - be - gone
Fix test
Move plugin to be a lister ( )
Disable sum test
[ DOCS ] Add . NET class tip ( )
Build : Switch to fedora - 25 - test for windows .
test : fix IndexPrivilegeTests after removal of predefined privileges
[ ML ] fixing integration tests for bucket ( )
Marvel : Make sure to corrupt referenced tlog files when executing transport actions
Watcher : Ensure triggered watch is picked by default
Remove garbage
fix typos in percolate docs
Query DSL : Allow to default certain settings in query_string / field , closes .
[ DOCS ] Note ` index ` module ( )
move to implement ` 2 . 0 . 0 . Beta1
Merge pull request from cbuescher / feature / query - refactoring - constantscore
Docs : Add Hebrew analyzer
[ CCR ] Add create_follow_index ( )
Small test installation ( )
javadoc sample name to 2 . 3
Add unreleased v6 . 2 . 3 version
[ DOCS ] Remove beta qualifiers from metricbeat . asciidoc ( )
Merge pull request from rjernst / buildsrc_test_cleanup
addressed various comments
Fix Docker packaging tests on Windows ( )
[ TEST ] Add missing sort failure to HasChildQueryBuilderTests
[ ML ] Improve handling of memory field value limit ( )
remove the nocommit
Merge pull request from polyfractal / bugfix / movavg_double_predict
Alerts : Forgot to prevent the AlertsClient to determine the explicit index .
[ rest ] multinode cat - level REST test
Upgrade to lucene - 7 . 0 . 0 - snapshot - a0aef2f
SQL : Fix column size for IP data type ( )
Merge pull request from nik9000 / reindex_wait_for_completion
[ Watcher ] Update to elasticsearch 1 . 0 . 0
Merge pull request from jasontedor / remove - CONF_FILE
[ DOCS ] Enabled code snippet testing for start datafeed API ( elastic / x - pack - elasticsearch )
test : makes use AwaitsFix in test
Build : Move REST rest test to core ( )
Async search processor should not be updated
[ DOCS ] Update enrich APIs to docs ( )
remove unused code
[ TEST ] Kill documents into an index response package via a single segment
Remove DiscoveryNode # masterNode in favour of existing DiscoveryNode # getVersion
Removed unused code
Changed ` index . auto_create_index ` to ` useradd
[ ML ] Minor the ML plugin to analytics ( )
Disallow runtime runtime fields ( )
Upgrade to lucene - 7 . 0 . 0 - snapshot - a0aef2f
[ ML ] Fix bug in MlStrings # hasValidLengthForId
[ DOCS ] Fixed typos
Reworking tests for cluster alias in mixed cluster ( )
Removed unused code
nicer error message when restoring an index
Aggregations : Significant Terms Heuristics now into account if there are no longer exist
Merge pull request from s1monw / expected_shard_size
test : don', "'", 't randomly prepare exception for RemoteIndexAuditTrailStartingTests
Remove binary type in ShrinkAction ( )
[ TEST ] Fix compilation
Increase timeout of cluster state setting ( )
Merge pull request from rjernst / fix / 10659
fetching retention fields without fields without fields .
Re - enable BWC tests after backport of mapping ( )
Fixing roles', "'", 's changes on the docs ( )
Test : Make run list more obvious
Merge pull request from s1monw / handle_source_canceled
Adds more support for ingest processor types ( )
[ DOCS ] Fix typo in RoutingNode comment ( )
change local of the index to', be 'in the ClientSSLService
Add ParsedCardinality ( )
Merge pull request from jasontedor / settings - cleanup
[ Security ] Fix tests for ctx . get . transform ( ) ( )
Docs : Fixed the link
Live primary shards before cancelled primary shard ( )
[ DOCS ] Add missing icons to transform field examples ( )
Added version 2 . 0 . 0 to WatcherVersion
Local : cleanup cleanups
Tests : Fix randomizedtest fail
[ Tests ] Check QueryProfileShardResult should be null ( )
[ TEST ] Add . * = size / mapping issues
QL : Wrap ( )
[ DOCS ] Note cat index API ( )
Do not timeout in test when master is gone ( )
Rename CCR ( )
[ TEST ] Add IndexShardOperationPermits to DistributorDirectoryTest
Added the API definition for ` _cache / clear `
Mute test
[ TEST ] Fix newline in PutIndexTemplateRequestTests ( )
Merge pull request from nik9000 / xlint2
SQL : Fix parsing of CAST and type ( )
Fix certutil error
Transport : added back support
Wait for guice to be snapshot before running ( )
Raise an default setting
Merge pull request from talevy / more_details_factory_factory
Marvel : Mute IndexStatsIT
Merge pull request from cbuescher / feature / query - refactoring - spanmultiterm
Core : only use groovy wrapper for bulk recoveries
[ DOCS ] Remove X - Pack installation section ( elastic / x - pack - elasticsearch )
Merge pull request from s1monw / renew_syn_on_merge
Fix InternalAutoDateHistogramTests # hasSource ( )
removed javax . mail from test dependencies
Merge pull request from rjernst / fix / 12329
add a BridgePartition to generate empty keyword parameter
Fix issue in RangeAggregator
Merge remote - tracking branch', "'", 'origin / master', "'
Minor DocWriteResponse in FieldSortBuilder ( )
Use ephemeral ports for ESSingleNodeTestCase ( )
fix compile warnings
[ Test ] Fix RepositoryURLClientYamlTestSuiteIT
disable value of field_caps requests ( )
Merge pull request from cbuescher / feature / query - refactoring - spanmultiterm
Remove unused code
Bulk processor and policies ( )
[ DOCS ] Update ILM docs to plugin . asciidoc ( )
Merge pull request from jdconrad / init2
Painless : Decouple PainlessLookupBuilder and Whitelists ( )
[ Tests ] Don', "'", 't rename indexShard .
[ ML ] Remove record_count for ML mappings ( elastic / x - pack - elasticsearch )
Better logging for transport client permissions
properly make sure start is used only when active isn', "'", 't exist
add a ability to control the netty cluster state request
[ DOCS ] Streamline GS search topic . ( )
[ DOCS ] Re - Pack training_percent limitation . ( )
HLRC : Use Optional in validation logic ( )
move abort of bulk requests , closes .
[ DOCS ] Add tab widget for transform related APIs ( )
Merge pull request from jasontedor / nested - index - delete - nocaptures
[ DOCS ] Add shared node settings to ES docs ( )
Merge pull request from rmuir / spi_reload
[ TESTS ] Implement Tombstone REST tests
Remove the MapperBuilders file ( )
Merge pull request from kkirsche / patch - 2
SQL : Fix issue in UnresolvedRelation
[ DOCS ] Reformats cat API API docs ( )
[ TEST ] add afterDistribution timeout to expectedTimeToHeal
_nodes / optimize
Change IndicesAdminClient # existsAliases to IndicesAdminClient # aliasesExist .
Fix testThatNonExistingTemplatesAreAddedImmediately ( )
[ TEST ] Don', "'", 't rely on ` new Date ( elastic / x - pack - elasticsearch )
Merge pull request from rjernst / normalize_plugin_zip
Add a ` debug ` settings ( )
Merge pull request from tebriel / plugin_docs
Remove PreProcessModule
Remove traces between connect ( )
Update 1 . 2 . 3 release
Fallback to node if transport profile is empty
[ Security ] allow FollowIndexAction . Request class ( )
Add ingest processor for ingest processor ( )
[ TEST ] Fix compilation
better logging message for fs / index', errors ', closes .
[ TEST ] Only if port is smallish
[ DOCS ] Add xref to multiplexer query docs ( )
[ TEST ] fix tests for ctx . _ttl more leeway
Merge pull request from jasontedor / remove - CONF_FILE
Merge pull request from rmuir / nukeSettings
Switch breaking changes to breaking changes
Remove garbage
[ PARSER ] Calculate Alder32 Checksums tests
Removed AggregationContext . cacheRecycler ( ) instead of SMILE
allow to set the index level cluster state , ` node ` and ` kstem ` , closes .
[ DOCS ] Add missing comma
[ DOCS ] Fix dup typos in 8 . 0 breaking changes ( )
Simplify AuthorizationService in groovy core
[ TEST ] Protect UpgradeIT from IntervalSchedule to CronSchedule ( )
[ TEST ] Make sure that IndicesRequestTests is repeateable ( elastic / elasticsearch )
HLRC : Add parameters to stopRollupJob API ( )
[ ML ] Fix online updates . analytics fields ( )
[ Tests ] Fix CcrRetentionLeaseIT . testRetentionLeaseIsRenewedDuringRecovery ( )
Add unreleased v6 . 2 . 3 version
SQL : Make SqlAction tests more in sequences ( )
EQL : Introduce support for sequences ( )
Adding more pressure header
Mapper : An from CamelCase name , closes .
rpm . txt can never be empty

add repo and license
Merge pull request from droberts195 / grep_minus_e_portability_fix
[ TEST ] Harden InternalExtendedStatsTests ( )
Use a separate build directory for Eclipse .
[ DOCS ] Update readme for upgrade ( )
Fix prerelease asciidoc docs
SQL : Fix issue with UnresolvedRelation
[ TEST ] Add test for MockRepository , and checks for termination
Use reindex rest - api tests for rest test ( )
Merge pull request from tebriel / plugin_docs
AsyncSearchMaintenanceService should not delegate the scheduler ( elastic / elasticsearch )
Docs : Remove the welcome page from the docs ( )
[ DOCS ] Add the few get API
[ TEST ] Pass a clientSSLContext in LdapTestCase
Move gradle property check to global build info ( )
Cleaned up the bytes on the review
Use a comment block to comment out release notes
[ TEST ] add additional logging to testCorruptTranslogFiles
Remove settings for shard allocation settings ( )
[ TESTS ] Licenses should not rely on other nodes
improve filtering on search requests , closes .
Gender neutral README
Add HLRC processors ( )
[ DOCS ] Documented Metricbeat ECS logs ( )
add a debug logging
Azure repository : Accelerate the listing of files ( )
[ ML ] [ Data Frame ] fix test for continuous jobs ( )
[ DOCS ] EQL : Document ` size ` function ( )
Adapt to review', 1 'to upstream changes .
use atomic for elasticsearch - test dependency
simplify plugin for checksums ends permissions
[ DOCS ] Linked cat - pending to cluster - pending
Run eclipse files
Remove unused method Analysis # isNoStopwords .
Add Cumulative Cardinality aggregation ( )
painless : Painless PainlessLookupBuilder and Whitelists ( )
add IndicesAdminClient Cache to nodes , closes .
Add integration tests
TEST : Muted IndexStatsIT
[ TEST ] HttpCompressionIT is a real IT
Merge pull request from rmuir / painless_compile_exceptions
Remove cruft
Docs : Remove docs that uses index . ( )
SQL : Add toXcontent test for SqlResponse ( elastic / x - pack - elasticsearch )
[ DOCS ] Fix links docs ( )
SQL : Add catalog concept for SqlResponse ( )
Support agg of agg requests in the agg ( )
Merge remote - tracking branch', "'", 'upstream / master', "'
Merge pull request from rjernst / ingest_plugin_api
SQL : Make Literal on NamedExpression ( )
SQL : Remove unused method from SQL ( )
Mappings : Don', "'", 't allow try for concrete mapping name
Remove ` include_type_name ` in favor of ` levenshtein ` ( )
[ ML ] Mute data frame analytics BWC tests ( )
Mapper : merging
allow to control term query for a field cache , closes .
Fix compile issue with xor processor values ( )
Fix java6_u31 compile error w . r . t . indices
rpm can try data - 8 natively ( )
first cut at integrating new settings infra
disallow deleting lifecycle policies that are referenced by existing indices ( )
Fix compilation in Ecipse ( )
Merge pull request from s1monw / move_refresh_into_index_service
Update indices after backport ( )
Build : Added Bloodhound
Add field to field_value_factor
Remove unused import
security : Merge ClusterInfoCollector in ClusterStatsCollector
CONSOLE - ify filter aggregation docs ( )
Aggregations : # . yml : . blocks doesn', "'", 't may handled .
Switch AggregationSpec to ContextParser ( )
In ctor in GeoPointFieldMapper ( )
[ DOCS ] fixed docs for search . asciidoc
Add start_basic
Merge pull request from rjernst / buildsrc_ide
Add v6 . 8 . 0 version
Sorting on _score in the URI aggregation ( code
Set neutral support
[ DOCS ] Add PGSync to setup pages ( )
[ Monitoring ] Remove Dedicated IndicesStatsCollector ( elastic / x - pack - elasticsearch )
Rename bin / settings for bin / elasticsearch
Added support for acknowledgements to api
[ Kerberos ] Remove builders from RollupJobConfig ( )
Depend on elasticsearch - parent
[ DOCS ] Add PIT as community - level integration ( ) ( )
Merge pull request from rjernst / jigsaw
[ TEST ] remove needless ClusterScope annotation from NettyTransportMultiPortTests
Marvel : Add support for Ctrl - Close event on Windows platforms to gracefully shutdown node
Test : More attempts at forcing netty test to always use netty
Merge pull request from nik9000 / remove_index_buffer_size
Migrate', gradle 'REST tests to 6 . 8 ( )
[ TEST ] decreased number of replicas to 0 in refresh tests
Merge pull request from elastic / fix / compressedstring_equals
[ DOCS ] Fix typo in ingest processor docs ( )
Use comparator for Reconfigurator ( )
update to 0 . 12 . 0 SNAPSHOT version
Split NestedQueryParser into toQuery and formXContent
[ DOCS ] Move rollup APIs to docs ( )
Removed ParseFieldMatcher usages from SearchService
Add more asciidoc for the index aliases API ( )
Docs : indices that now that the geographically is not present
Merge pull request from cbuescher / feature / query - refactoring - spanmultiterm
Remove BufferedDeletesStream
Merge pull request from ywelsch / fix / handling_index_state
Remove SecurityTransportModule
Add ` create_doc ` and ` _source_exclude ` HLRC ( )
Bad asciidoc link
Remove code 2 . x test from master
[ DOCS ] Add ML overview details ( )
[ DOCS ] Add docs for bucket_span responses ( )
Optimize the order of bytes in uuids for better ( )
Change GetFieldMapping API to broadcast requests ( )
Test : allow all test for mapping data tests
Added 1 . 3 . 0 to WatcherVersion
[ DOCS ] Update documentation for the Go plugin page ( )
Use', parameters 'for cleaning index time ( )
Merge pull request from jpountz / fix / compressedstring_equals
[ TEST ] use assertAcked when creating indices
Security : IndexAuditTrailTests use of ` htmlStrip ` field .
[ ML ] fixing DFA jobs with data frame analytics ( )
Merge pull request from cbuescher / feature / query - refactoring - constantscore
[ TEST ] mute REST tests that uses it
Add RamUsageEstimator # sizeOf ( ) ( )
Add mappings
[ Docs ] Fix typo in jump context docs ( )
Upgrade to Lucene 8 . 4 . 0 - snapshot - 08b8d116f8f ( )
Fix ` MapperService . unmappedType ` to not raise ` index . settings ` ( )
[ DOCS ] Fixed typos in ` ShardRouting ` API docs ( )
Relocation targets ( )
Merge pull request from rmuir / crazy_ubuntu
testRecoveryAfterPrimaryPromotion shouldn', "'", 't support comma to field query .
Aggregations : Fix percolator aggregation docs
[ DOCS ] fixed query authorization links ( )
addressed review comments
smart Query : add ` reverse_cout ` and ` reverse_term ` to ` order ` options , closes .
Merge pull request from jasontedor / strings - split - as - array
SQL : Fix issue with WEEK / ISO_WEEK / DATEDIFF ( )
inner_hits :', add 'generics in Terms API
properly ask for fs node close in 1 . 5
Merge pull request from jimferenczi / max_merge_settings
Merge pull request from polyfractal / feature / movavg_double_predict
Add UBI docker builds ( )
[ DOCS ] Remove beta label from normalizers ( )
Remove SecurityTransportModule
when doing the root cluster state in rest tests
Mute WatcherRestIT 20_minimal_body
Merge pull request from rmuir / jail_filestore_exception
move to mvel 2 . 1 . beta1 ( versioning changed from 2 . 1 . rc1 )
Merge pull request from jpountz / upgrade / simulate_index_type_not_required
Fix parsing for _all aggregation ( )
Remove DiscoveryNodes # localNode in favour of existing DiscoveryNodes # getLocalNode
[ DOCS ] Rename update APIs to use composable settings ( )
Use preferIPv6Addresses for Reconfigurator ( )
when a routing shard to be applied when a shard allocation is changed .
Fix test bug
switch to plugin pom
Make peer recovery pass
Document testing for elasticsearch - parent
GeoContextMapping validates references to AbstractStreamableXContentTestCase ( )
Mappings : Lock down _uid field
Improve logging permissions
Use Lucenes expectThrows ( ) when parsing exceptions
Correct merge filter filter in terms filter ( )
Fix indentation
[ TEST ] [ Tests ] [ Tests ] Also invalid StreamInput / nightly test to expectedTimeToHeal
[ Rollup ] Remove builders from RollupJobConfig ( )
make transform query serializable
Fix testSeqNoCollision ( )
Tie coordinating search context should shard shard logging ( )
upgrade to guava 12 . 0 . 1
Merge remote - tracking branch', "'", 'upstream / master', "'
[ DOCS ] Streamline GS search topic . ( )
Remove', ['Use committed annoation to MessageDigest . reset
Remove GceMetadataService interface from IndexShard
Include ingest processor in the docs ( )
[ DOCS ] Reformats replacement for node stats API ( )
EQL : Introduce catalog scripts to ScriptDocValues ( )
[ Docs ] Added missing word to field query
allow to be ToXContentObject the new nodes
Analysis : Add support for Ctrl - Close event on Windows , closes .
SQL : Switch for verification_exception with IP files ( )
Fix get get api tests
[ TEST ] Check accounting breaker for all test in InternalEngineIntegrationTest
[ TEST ] Add back accidetially removed test logging
[ DOCS ] Call Copyright notices to build in Elasticsearch docs ( )
[ TEST ] Mute IndicesCustomDataPathTests
SQL : Make Literal for NamedExpression ( )
Adjust bwc tests after backport of ( )
Fix testThatNonExistingTemplatesAreAddedImmediately to use createLeaderIndex ( ) helper method ( )
Fix testSeqNoCollision ( )
Merge pull request from ywelsch / fix / index - cs - on - routingresult
Merge pull request from synhershko / patch - 6
Merge pull request from MaineC / enhancement / 8964
Remove redundant modifier
Cherry pick tests and seqNo recovery hardning ( )
: fix the rest tests that is rolled
Merge pull request elastic / elasticsearch from rjernst / search_parser
[ DOCS ] Move model APIs to docs ( )
[ DOCS ] Reformat ` keyword_marker ` token filter ( )
Build : Fix repositories detection for plugins ( )
[ ML ] [ Inference ] Unify top_classes detection time ( )
review comments
HLRC : Add machine Lifecycle API ( )
Fix in Translog # factor
Merge pull request from jasontedor / joiner - be - gone
Merge pull request from droberts195 / grep_minus_e_portability_fix
Synonym histogram facet
Add cliSetup filtering to index settings ( )
Upgrade to Jackson 2 . 7 . 0
Merge pull request from polyfractal / bugfix / movavg_double_predict
Remove PROTOTYPE from ExtendedBounds
Adapt to implement Writeable rather than Streamable
clean lastGc
[ TEST ] fix HDRPercentilesTests . testScript_ExplicitSingleValued_WithParams
Cleanup ClusterServiceIT # testClusterStateBatchedUpdates
[ TEST ] Mute IndicesCustomDataPathTests
LicensesService : Disable CName in org . elasticsearch . common . compress cleanup
protected when extracted / automatically default to false
Include warning of all missing test tasks ( )
include_in_all not overridable the index creation , closes .
make node closed settings serializable
[ TEST ] Revert accidential massive iterations
Sorting on _score in the cluster state , closes .
Merge pull request from rjernst / fix / 10659
HLRest : add security authenticate API ( )
Mute failing test
Merge pull request from gmarz / bugfix / movavg_double_predict
[ TEST ] Make sure there are repeateable
Packaging : Fix randomizedtest fail
[ DOCS ] Add shared info to security book .
Remove cruft
Unzip quietly while provisioning virtual machines
Mute failing test
add the ability to control the cluster state logging
[ TEST ] Use private useage of query simple query tests
Merge pull request from tebriel / plugin_docs
[ TEST ] remove double field check in docs ( )
[ DOCS ] Relocate Shadow filter docs ( )
Fixes review calc api
[ ML ] Ensure total hits are tracked to thrown ( )
Merge pull request from s1monw / renew_syn_on_merge
Merge pull request from rjernst / save_eclipse
Introduced valueForString in TransportClient et . al
allow to set the default date math by default
Remove garbage
[ TEST ] restored gateway lifecycle created by a single cluster pool
[ ML ] Add create_follow_index param to integration tests
Merge pull request from s1monw / feature / ingest
lucene 4 : Upgraded ContentIndexSearcher
Remove unused clear code
Minimum version hidden setting ( )
Update TcpHeader tests after backport ( )
Merge remote - tracking branch', "'", 'origin / master', "'
[ TEST ] Mute IndicesCustomDataPathTests
Added plugin for plugin . xml
Improved documentation with documentation with new promotion
Update to Elasticsearch 0 . 90 . 1
[ TEST ] only not await all the comparison
[ TEST ] Revert accidential massive iterations
test : add missing test to', ['Refactor
[ ML ] Improve serialization in DF analytics mappings ( elastic / x - pack - elasticsearch )
[ DOCS ] Changes experimental flag to beta in DFA related docs ( )
Remove some unnecessary methods from SearchContext . java
Mute WatcherRestIT 20_minimal_body
[ DOCS ] EQL : Add search and open index - time params ( )
Fix compilation statement
handle an async index to index settings
[ TEST ] packaging : enable trial license
Add an alias appendable / plugins ( )
MINOR : Cleanup Dead Code in Painless Package ( )
SQL : MATCH to new the QUERY time on an array ( )
Cleanup range field tests . ( )
doc
Bad indentation link
HLRC : Add support for put user API ( )
Docs : Reorganized multi - node attribute
LicensesClientService . register works should not be mappings ( )
Merge pull request from rjernst / normalize_plugin_zip
delete unused XContentStructure class
Mute IndexShardTests # testDocStats
[ DOCS ] Add impact - community - node ( )
Corrected comment grammar
Merge pull request from rjernst / integ_test_deps
Added circular imports
Alerting : adjusted visibility to package the cluster health execution
add a debug logging
Remove PreProcessModule
Merge pull request elastic / elasticsearch from rjernst / rest_alone
Mute TestClustersPluginIT . testIdempotencyIsEnforced
Add docs for cluster . remote . connect ( )
[ DOCS ] Re - enabled include official - supported ( )
[ CCR ] Implement CompositeIndicesRequest
Basic derivative reducer
Internal : Use the translog ( in the new balancer ) , not preferIPv4Stack
Upgrade to Lucene 4 . 10 . 2
[ TEST ] Add SuppressSysoutChecks to DistributorDirectoryTest
Docs : Documented cluster allocation filtering
Scripting : Rework KV Processor ( )
Use comparator for an index
update version to 1 . 0 . 0 - Beta1
Merge pull request from rjernst / save_eclipse
Merge pull request from s1monw / fold_in_index_cache_module
update to 0 . 12 . 0 SNAPSHOT version
Watcher : Use Optional in mustache smoke tests
prepare release elasticsearch - analysis - stempel - 1 . 0 . 0
LicensesClientService . register . disable_purge should not be empty
remove PROTOTYPE from ExtendedBounds
fix typo in docs ( )
Merge pull request elastic / elasticsearch from rjernst / license_state_removal
Make sure that issue is repeateable ( RestStatus . INTERNAL_SERVER_ERROR )
Wildfly fix use the rest high level client ( )
Tests : Add REST tests
Painless : Add trap tests
Implement new . ml - anomalies ( )
fix GoogleCloudStorageHttpHandler ( elastic / x - pack - elasticsearch )
Fix Infinite Retry Loop in loading RepositoryData ( )
Cleanup ClusterServiceIT # testClusterStateBatchedUpdates
Disable java and maven version checking
Merge pull request from rjernst / buildsrc_test_cleanup
Merge pull request from jasontedor / settings - cleanup
Docs : fix docs for the upgrade API ( )
Added missing SystemD API for typed_keys
provide code
[ DOCS ] Add PIT to search level docs ( )
[ DOCS ] Add redirect for heap size ( )
Merge pull request from llasram / hll - estimate - bias - nn - k - 6
Merge pull request elastic / elasticsearch from rjernst / deguice1
test : don', "'", 't randomly the test with the same shard .
Renew GoogleCloudStorageHttpHandler ( )
[ DOCS ] Add abbreviated titles
Packaging : Fix randomizedtest fail
[ ML ] Fix grabbing bwc tests
Shield Docs : Fixed bad link
Remove types for controller location ( )
Adressing review comments
Update to Lucene 4 . 10 . 2
Remove PROTOTYPE from ExtendedBounds
Enable customizing REST tests for backport ( )
Create es - 1 . 2 branch
Clean up index alias tests ( )
Remove GceMetadataService interface as not needed
remove cruft
Merge pull request from rmuir / jdkhell
Don', "'", 't load BytesRef in ThreadLocal
Cherry pick tests and seqNo recovery hardning from
[ TEST ] remove outdated TestLogging annotation
prepare release elasticsearch - cloud - aws - 1 . 14 . 0
add the ability to wait for nodes are green
Remove _create ( elastic / elasticsearch )
[ TEST ] decreased number of replicas to 0 in refresh tests
[ BUILD ] Fix RepositoryURLClientYamlTestSuiteIT
[ DOCS ] Move API API ( elastic / x - pack - elasticsearch )
doc
Fix BytesStreamInput .
[ DOCS ] Reformat ` remove_duplicates ` token filter ( )
update 7 . 1 . 0 release
inner_hits : Fix IndexRecoveryCollectorTests
Docs : Fixed bad link
Use new translog and field in query statement
update documentation with release 2 . 4 . 1
Merge pull request from rmuir / nukeSettings
[ TEST ] Ensure reflectively is repeateable ( elastic / x - pack - elasticsearch )
Merge remote - tracking branch', "'", 'upstream / master', "'
[ DOCS ] Fixed link to the api docs from the API . ( elastic / x - pack - elasticsearch )
Fix CreateRepository Requeset in deprecation condition docs
[ TEST ] Fix failure to ensure setup
Test : More test looking on Windows ( )
Update to AWS SDK 1 . 7 . 13
fixed compilation errors due to upstream changes in elasticsearch itself
Adapt to yaml 5 . 0 . 0 - Beta1
Remove transport transport transport from transport transport mode .
upgrade to latest aws jdk , closes .
Always use soft - deletes in InternalEngine ( )
Fixed typo
elasticsearch the R . parent
Remove extraneous CreateIndexRequest ( )
Fix comments in SequenceNumbersService . java
Merge pull request from rmuir / jdkhell
[ HLRC ] Add level open memory APIs ( )
Update search - filter . asciidoc ( )
Merge pull request from s1monw / jdkhell
Merge pull request from nik9000 / issue10371
[ DOCS ] Fix typo in jump link . ( )
Merge pull request from jasontedor / remove - CONF_FILE
[ DOCS ] Fix typo in context suggest
Mute WatcherRestIT 20_minimal_body
Mappings : Remove mapper listeners
Merge pull request from s1monw / jdkhell
Fix Memory Leak tests
Make PercolatorFieldMapper aborts work
Split NestedQueryParser into toQuery and formXContent
Add CONSOLE to delete - based docs
Add SAML IdP check for netty state action ( elastic / x - pack - elasticsearch )
Merge pull request elastic / elasticsearch from rjernst / rest_alone
Overwrite ignore_above parameter when creating a plugin
 )
[ TEST ] fix CommonTermsQueryBuilderTest
better handling of shards of the index
Added support for the include_type_name ` terms ` to ` _index ` and ` max_value ` .
Tests : Increase all tests that are referenced , but
Fixed issue in search . allow_expensive_queries ( )
upgrade to Lucene 3 . 6 . 1
[ TEST ] Add SuppressSysoutChecks to DistributorDirectoryTest
[ TEST ] fixes from IntervalSchedule to CronSchedule .
Fix TimeSeriesLifecycleActionsIT . testRolloverAlreadyExists ( )
Remove unused constant
lucene 4 : Upgraded ContentIndexSearcher
Merge pull request elastic / elasticsearch from PhaedrusTheGreek / group_search_noattrs
[ DOCS ] Enabled code snippet testing for start datafeed API ( )
[ TEST ] fix HDRPercentilesTests . testScript_ExplicitSingleValued_WithParams
Remove heuristics that uses TransportSingleCustomOperationAction . ( )
Add REST API to high level REST API ( )
Fix CoordinatorTests ( )
Unmute rest test out of 2 . 2
Fix Broken Math in NodeConnectionsServiceTests ( )
Deprecate types from timestamp_field types ( )
Fixed typos in examples .
[ TEST ] Randomize FS Translog implementation
[ DOCS ] Reformat ` remove_duplicates ` token filter ( )
Merge pull request from jdconrad / deprecate
[ BUILD ] Fix RepositoryURLClientYamlTestSuiteIT
Remove unused code
wrap test
[ DOCS ] Rename description to the Go experimental ( )
Update to elasticsearch 1 . 0 . 0
Merge pull request from jasontedor / call - split - split - only
Enable Gradle gitignore for the separate jar ( )
[ DOCS ] Add information about security security snapshot ( elastic / x - pack - elasticsearch )
[ CCR ] make sure auto - follow open " ( )
[ ML ] Better detection of binary input in find_file_structure ( )
Add scroll index to index stats
fix test stability
fix test bug
Merge pull request from ywelsch / fix / compressedstring_equals
Mappings : Remove mapper / resident caches .
remove commons - by as', unused 'field
Merge remote - tracking branch', "'", 'origin / master', "'", 'into json_strict_duplicate_checks
[ TESTS ] Licenses should using only for one recovery
[ DOCS ] Remove beta qualifiers from node . asciidoc
Docs : Update plugins . asciidoc
Merge pull request from jpountz / fix / compressedstring_equals
Adapt to current the remote cluster ( )
removed unused code
Added support for objectCount and shardsIt
Count / Indices when destination ( )
Remove logging from TermVectors docs
[ DOCS ] Clarify title for heap settings ( )
[ DOCS ] Fixed typo in jump link . ( )
Search : Search requests hangs when providing fields , closes .
Remove peers from ShardStateAction ( )
[ DOCS ] Add note about long to the page for the profiler
doc
Remove registerQueryParser
Update to elasticsearch 1 . 4 . 0
Merge pull request from mrsolo / features / thirdparty
Fix error message in AzureTestUtils
lucene 4 : Converted InternalIndexShard # checkIndex .
Merge remote - tracking branch', "'", 'origin / master', "'
Build : Fix eclipse jar for zip input ( )
Remove references to mapping type in FieldTypeLookup . ( )
[ DOCS ] Remove beta label from normalizers ( )
Remove PROTO of tuple ( elastic / elasticsearch )
Remove dead modifier
Merge pull request from javanna / enhancement / terms_lookup_query_builder
[ DOCS ] Compacted enrich index API docs ( )
If all percolate api
Improve Error Handling in SearchableSnapshotAllocator ( )
Add comment to S3 realm values ( )
[ TEST ] fix CommonTermsQueryBuilderTest
Mute FullClusterRestartSettingsUpgradeIT
Do not warm - on a test that occur
[ DOCS ] Add data streams to node APIs ( )
[ TEST ] Add SuppressSysoutChecks to DistributorDirectoryTest
[ DOCS ] Add xref to multiplexer query ( )
Replace several try - finally statements to be enabled
change the root type to the mapper mapping , closes .
Added the API definition for the put index
Fix InternalSearchHit # Murmur3FieldMapperTests
[ Docs ] Update link in removal_of_types . asciidoc ( )
optimize encrypted documents
[ TEST ] Add missing one class to get REST tests
Fix path for path . 14 . 0
[ TEST ] use assertAcked ( ) and wait for termination
[ TEST ] Fix test bug in MultiOrdinalsTests .
update plugin to 1 . 4 . 0 - snapshot - e648d601efb ( )
Fix missing rest parts for search . allow_expensive_queries ( )
Mute FullClusterRestartSettingsUpgradeIT
[ DOCS ] Chunk depth now configurable ( elastic / x - pack - elasticsearch )
[ TEST ] fix CommonTermsQueryBuilderTest
Mappings : Extend KV
Revert release elasticsearch - cloud - aws - 1 . 14 . 0
Do not make accurate fields ( )
[ DOCS ] Reformat ` keyword_marker ` token filter ( )
Remove the MapperBuilders from the help ` id ` element .
Remove translog timeout in shouldFlush
blacklist forbidden integration tests again ( Closes ) ( )
Add a typo about doc values in the typo . ( )
Reverse from renaming ( elastic / elasticsearch )
[ ML ] Remove Dedicated IndicesStatsCollector ( elastic / x - pack - elasticsearch )
[ DOCS ] Fix " NamedID " ( )
Mute failing test ( )
[ TEST ] Fix indexed test issues
Mute CcrRetentionLeaseIT . testForgetFollower
[ DOCS ] EQL : Document ` size ` limit for pipes ( )
Setup some memory data
[ Docs ] Add missing word in nested suggest
[ ML ] Enforce the stop datafeed aoi to wait for an index that fails ( )
only ask for the node node request has', been 'required .
doc
[ TEST ] Simplify IndexAliasesTests
rename parsing sig
[ DOCS ] Fix typo in RoutingNode ( )
Merge pull request from jpountz / feature / bool_fields
Merge pull request from rmuir / nukeSettings
Fix log
Replace explicit unittests pass
Split NestedQueryParser into toQuery and formXContent
Remove unused constant
Added the API definition for ` _cache / clear `
[ Kerberos ] Remove unused Request apis
Local Gateway : cleanup shard service to be stored under each shard , closes .
add a separate management
Fix comparator .
Added missing examples
Prevent ordinals in the completion field data
 "'", 't check with empty thread pool info ( )
[ Docs ] Add missing word in nested . asciidoc ( )
[ DOCS ] Add setup troubleshooting details ( )
ignore intellij project files
Data stream support for get field agg API ( )
 )
Search - on requests should also shard data changes
Update maven assembly plugin to latest version : 2 . 3
Fix test that is merged in the wrong mapping .
Query refactoring : query_string
[ ML ] Remove Dedicated IndicesStatsCollector ( elastic / x - pack - elasticsearch )
[ DOCS ] Add missing comma
[ DOCS ] Fix typo in get suggest
[ ML ] Remove partition_field param from GetBucketsAction ( elastic / x - pack - elasticsearch )
Remove GceMetadataService interface as not needed
[ DOCS ] Add impact sections to the get / policy docs .
Upgrade to Lucene 4 . 10 . 2
Include quartz logging on applier license tests
Add old index validation ( )
[ ML ] fixing datafeed serialization for analytics ( )
[ DOCS ] Fiexes length links to node . asciidoc ( )
Remove unused code
HttpInfo to implement Writeable rather than Streamable
Query refactoring : query_string
[ Docs ] Update length - tokenfilter . asciidoc ( )
Merge pull request from talevy / ingest - fix - nocaptures
Merge pull request from rmuir / spi_reload
move to set the node thread pool
[ DOCS ] Update readme for Asciidoctor migration ( )
[ ML ] Reuse intervening # staleTask ( . . . ) to wait for job and datafeed status and
apply review comments in IndexShard
[ DOCS ] Add delete snapshot to ping indices docs
Core : Add bulk api implementation to the explain bulk API .
wrap update upsert used with the main docs case
Configuration : Add ability to configure alerting by creating . alerts / config / < name > documents in the index .
SQL : Don', "'", 't allow inexact fields for MIN / MAX ( )
For not CircuitBreakerStats to implement Writeable rather than Streamable
[ Tests ] Update tests for backport ( )
Marvel : Add a integration test for xplugins
Merge remote - tracking branch', "'", 'origin / master', "'
[ TEST ] fix IndexAliasesTests
provide lastGc
Fixed hunspell typo in exception
add an API to delete operation to source and source doc as well
[ TEST ] Skip double null check in TransportSearchActionTests
Aggregations : Add cumulative sum aggregation ( )
[ ML ] Mute JobProvider . testWatchdog
Depend on elasticsearch - parent
make sort usages serializable
remove unused objects .
inherit from BroadcastOperationRequest
Added the API definition for the percolator API spec
Merge pull request from jasontedor / remove - CONF_FILE
Make return recovery of MapBuilder # immutableMap Map
Remove @ Override utility method
[ DOCS ] Add missing icons to REST API ( )
 )
Add useful logging on error message in osprobe ( )
[ Tests ] Add test for search mapping
Add shuffling shuffling to plugins ( )
L Data frame - analytics snapshot up ( )
review comments
Fix P / GetRequest / UpdateRequest .
Add bwc tests for ValuesSourceType ( )
Remove unused interface Analysis # isNoStopwords ( )
[ DOCS ] Disable explicit index index docs ( )
Remove BufferedDeletesStream
Merge pull request from jpountz / fix / compressedstring_equals
Merge pull request from jasontedor / remove - CONF_FILE
hards rebalancing when forking processors ( )
prepare for next development iteration
Merge pull request from s1monw / fold_in_index_cache_module
[ TEST ] Remove potential ClusterScope annotation ( elastic / x - pack - elasticsearch )
[ TEST ] Revert accidential massive iterations
prepare release elasticsearch - analysis - stempel - 1 . 8 . 0
update headers
Adds a minimum interval to ` auto_date_histogram ` . ( )
Cherry pick tests from query_string query ( )
Tests : Force comma to Merge ( )
Add ` beforeIndexAddedToCluster ` callback
Corrected', "'", 'shape', "'", 'support for _cat / nodes
Fix tests
Remove unused mapStringsOrdered method ( )
Merge pull request from rjernst / simplify_bwc_path

Remove spaces from the docs ( )
[ TEST ] Fix test to not rely on a node .
[ DOCS ] Add note to community clients docs ( )
[ TEST ] use assertAcked ( )
Fix CoordinatorTests ( )
Docs : Update plugins . asciidoc
Add version 5 . 4 . 3 after release
Add a ability to data streams ( )
[ ML ] fixing intervening field field checks ( )
Tests : Make sure that watcher is never referenced ( elastic / elasticsearch )
[ TEST ] fix CommonTermsQueryBuilderTest
[ ML ] Calendar event actions ( )
Cleanup : Remove generics from check
Remove javax . mail ( elastic / elasticsearch )
[ TEST ] Don', "'", 't rely on ` new Date ( ) ` ( 2 )
Fix P / GetRequest / UpdateRequest .
fix ignoring node tests
Fix Creating NOOP Tasks on SNAPSHOT Pool ( )
always start the unicast license
[ TEST ] Add tests for ctx . _ttl test .
Build : Fix incorrect plugins related integ test ( )
Fix InternalSearchHit # testForceStaleReplicaToBePromotedToPrimary ( )
[ TEST ] SimpleNodesInfoTests . testNodesInfos didn', "'", 't wait for all cluster to form properly
add the ability to set license framework
Don', "'", 't check if index is present to prevent races
[ TEST ] Fix ClusterApplierService # testDateRangeInQueryString ( )
Remove and ban unmodifiableMap in cluster package
clean wording with match_all Query
[ TEST ] Don', "'", 't use acks for each cluster health ( elastic / x - pack - elasticsearch )
Docs : Removed unnecessary configuration logic
BytesReference filter to properly return docs if possible
security : don', "'", 't call prepare index for reads ( elastic / elasticsearch )
add comment to CustomNameResolver interface
prepare release elasticsearch - analysis - aws - 1 . 11 . 0
Installing Discovery', "'", 's unzip method
Remove PreProcessModule
Merge pull request from jdconrad / init2
[ ML ] Rename DataStreamDiagnostics to use ( )
support md5 and PendingClusterTasksResponse for dates , closes .
Disable BWC tests while merging
upgrade to lucene 8 . 0 . 0 - snapshot - 08b8d116f8f .
[ CORE ] Remove special TestLogging ( )
Prevent )
[ DOCS ] Update documentation for the Go analyzers ( )
Rename bin / plugin to _xpack / groovy
[ DOCS ] Add redirect for heap only ( )
HLRC : Add support for the searchable_snapshot client to a HLRC ( )
[ TEST ] Use SecurityClusterClientYamlTestCase ( )
Merge pull request from rjernst / deprecate_exception_hurter
Mute', changelog 'defaults ( )
[ TEST ] Remove SecurityClusterClientYamlTestCase ( )
Re - enable FlushIT on new test ( )
[ DOCS ] Update ` scripted ` token filter ( )
Merge remote - tracking branch', "'", 'origin / master', "'
Remove @ Override annotation
[ ML ] Hide partition_scores field at create ( elastic / x - pack - elasticsearch )
Update ingest - aggregation . asciidoc
Fix compilation errors in RangeAggregator
add spelling
Delete delete API does not create the same request
[ TEST ] Fix test bug in MultiOrdinalsTests ( )
Fix ( simple ) _query_string in QueryBuilders ( )
Alerting : Split alerting forecasts into packages
inner_hits : add support for shape type in query_string / FilterParser
Merge pull request from jpountz / fix / 12329
Add ES licenseHeaders to precommit
[ Transform ] disable BWC as preparation for ( )
[ DOCS ] Adds impact - plugin . es ( )
Wire up IpRangeAggregation to ValuesSourceRegistry ( )
Also mmap query requests
Merge pull request from rjernst / deprecate_exception_hurter
[ GEO ] Add system Redirect
HLRC : add support for the searchable_snapshot ILM action ( )
Fix UOE when extracting value is overuse ( )
Add favicon
Deprecate ` _type ` in simulate agg requests ( )
Merge pull request from jdconrad / init2
[ DOCS ] Add missing comma
Add support for auto_generate_synonyms_phrase_query in match_query query ( )
[ ML ] Remove unused " merging " ( )
[ TEST ] Update Manager
[ Test ] Introduce RepositoryURLClientYamlTestSuiteIT
Reenable Bwc Tests After ( )
Replace manual copying an array or collection with static methods ( )
Change unmapped token filter in FieldTypeLookup ( )
[ TEST ] make sure we close the correct logging in the tests
remove unused imports
Fix LoggingOutputStream for deleteBlob ( )
[ DOCS ] Fine - tunes training_percent option . ( )
move to mvel 2 . 1 . beta1 ( versioning changed from 2 . 1 . rc1 )
[ DOCS ] Reformat ` keyword_marker ` token filter ( )
[ DOCS ] Fix typos in Painless examples docs ( )
Add test for the new repo
Merge pull request elastic / elasticsearch from rjernst / license_state_removal
Added the API definition for the rest client spec ( )
[ TEST ] fix CommonTermsQueryBuilderTest
Fix compile errors due to upstream changes
[ DOCS ] Changes failed transforms limitation . ( )
Merge pull request from cbuescher / feature / query - refactoring - spanmultiterm
Build : Omit license and notice files in zip ( )
Fix Integration license test
SQL : MATCH and QUERY results for function ( )
Don', "'", 't try to use system jna for naming conventions
Merge pull request from rmuir / nukeSettings
[ DOCS ] Add missing comma
[ ML ] [ Inference ] allowing per - model licensing ( )
Remove the index . examples ( )
Fix visibility of PreConfiguredTokenFilter ( )
[ TEST ] replace IndexAliasesTests
Remove old tests for ` query_string ` .
Merge pull request from rmuir / cleanup_search_request
Merge pull request from droberts195 / grep_minus_e_portability_fix
Remove ` InternalTestCluster . unmappedType ( )
blacklist v5 . 0 . 0 - alpha2
Merge pull request from MaineC / enhancement / 8964
[ DOCS ] Fix typos ( ) ( )
Fix settings
Create es - 1 . 2 branch
Fixes certutil tool ( )
externalize how to index , closes .
SearchScrollRequest to implement Writeable rather than Streamable
[ PLUGIN ] Licenses should only update ( )
Merge pull request from brwe / remove - CONF_FILE
when providing shards , closes .
[ DOCS ] Fix X - Pack indices docs ( )
Remove the TODO connect in the docs
Merge pull request from s1monw / cleanup_search_request
Use Lucenes expectThrows instead of guice ( )
remove NoopOutputStream
Remove dead code
Merge pull request elastic / elasticsearch from rjernst / rest_alone
Inline NettyHttpChannel # getStatus
if code
Docs : Add Hebrew analyzer
fix Zero . testSerialization
Enable some gradle version
Modify load average formats
[ TEST ] Use explicit type in result files .
Add integration tests for renaming ( )
[ DOCS ] Add redirect for query query APIs ( )
[ TEST ] use assertAcked ( ) and return analyzer to expectedTimeToHeal
Remove soft - deletes that using the dateline ( )
[ ML ] Change job to be memorySizeSettting ( )
[ ML ] Rename Calendar to JobResultsProvider ( )
[ GEO ] Add a lot of clear docs in TermFilter
Mapper : automatic analyzer mapper for field mapper , closes .
[ DOCS ] Add query APIs to transforms query ( )
Clear when index is active , closes .
Remove support for search search resolution ( )
Merge pull request elastic / elasticsearch from rjernst / move_license
Add Infrastructure to bin - s3
remove NoopOutputStream
Make DeleteStep retryable ( )
Revert " Mute MinimumMasterNodesIT . testThreeNodesNoMasterBlock ( ) "
prepare release elasticsearch - analysis - aws - 1 . 14 . 0
Merge pull request from jasontedor / remove - CONF_FILE
[ TEST ] remove support for multiple mapping
[ TEST ] [ Tests ] Fixed invalid test names to expectedTimeToHeal
Adressing review comments
Test : Clean up tests
[ DOCS ] Add redirect for ILM templates ( )
Fix Translog # testWithRandomException ( )
Add more Origin ( )
testRecoveryAfterPrimaryPromotion : Mute NPEs with @ removed abstraction
Packaging : Fix randomizedtest fail
Add back for precommit
fix simple field names
Introduce unit tests for ValuesSourceType ( )
[ Docs ] Fix typos in ingest processor docs ( )
Remove unused ParseFieldMatcher usages from SearchService
[ DOCS ] Updated terms re wildcard support
HLRC : Add ability to put user API ( )
Merge pull request from rjernst / copy_to_dynamic_object
Add UBI docker builds ( )
Fix testAutoFollowing retrieval in IT
Remove comparator for Reconfigurator ( )
Recovery : don', "'", 't update the mapping with the closed thread
[ DOCS ] Fix typo in search api docs ( )
Increased match queries when using invalid indices from nested queries , closes .
improve test stability
Update ` version . asciidoc ( )
Randomly use a cached thread ( )
test : index documents with the same type and increase root log level
Fixed tests for info , ping and index
Build : Omit CName in plugins ( )
Merge pull request elastic / elasticsearch from rjernst / deguice1
Merge pull request from dadoonet / docs / modify - java - api - master
HLRC : Add Pipeline capabilities to Calendar ( )
[ DOCS ] Add ML jobs to ML links ( elastic / x - pack - elasticsearch )
Remove ParseFieldMatcher usages from SearchService
[ TEST ] fix for multiple indices messages in REST tests ( )
Remove StackOverflowError for TermsLookup ( )
[ TEST ] Change from IntervalSchedule to AbstractStreamableXContentTestCase ( )
Merge pull request from jasontedor / remove - matchall - norms
Fix classpath in qa + failure ( )
Initial jdbc and enable the plugin framework
test : add missing date math to blacklist
Move tests from core for core ( )
Fix packaging
HLRC : Add parameters to stopRollupJob API ( )
max bucket reducer and sibling reducer framework
[ DOCS ] Fixed typos in jump query docs ( )
Recovery : only cancel if filling gaps
[ ML ] add create_follow_index ( )
Correct indention in synonym docs ( )
[ TEST ] Remove timeout of indices . stats test setups
Do not create unused CRUD ( )
[ ML ] Improve detection of machine starting snapshot ( )
Wipe Snapshots Before Indices in RestTests ( )
[ TEST ] [ ML ] Fix assertion after starting df - analytics job ( )
Update to elasticsearch 1 . 0 . 0
Removed benchmark logging
Remove LocalTransport from ExtendedBounds
Replication Actions to support for gt , closes .
Add docs
[ TEST ] Fix test failure in CompletionSuggesterBuilderTests
Remove deprecated RealmConfig constructor ( )
[ TEST ] Added not version 5 . 0 . 0 . Beta1
Minor improvement to no_match_size highlight parameter and highlight tests
Snapshot / Restore to support for SSL discovery .
Merge pull request from spinscale / master
Added :
better failure against the host logging in the native manner
Remove DiscoveryNode # localNode in favour of existing DiscoveryNodes # getId
Fix license http test
Update cluster . asciidoc
Move settings info to use correct
[ DOCS ] Add xref to multiplexer API docs ( )
Cherry pick tests and seqNo tests
test : removed strict name
[ SPEC ] Fixed bad name in the GET API spec
[ ML ] Fix streaming failure
[ DOCS ] fix ` search . type ` type ( )
removed javax . mail from test dependencies
move to 1 . 2 . 0 snap
Add Infrastructure to Run 3rd Party Repository ( )
Implement DataFrameAnalyticsAuditMessage and DataFrameAnalyticsAuditor ( )
cache md5 computation for histogram facet
test : add rolling - level tests ( )
[ ML ] [ Inference ] Add FAILED messages for regression ( )
Core : Lock that plugins can', "'", 't be used .
cache default on request and facet to be cancelled
Generate typos in synonym docs ( )
[ TEST ] packaging : enable trial license
Persistent : assertion ( )
[ DOCS ] Documented ` node . max_buckets ` setting ( )
[ DOCS ] Fix typo in data stream API docs ( )
Aggregations : Add cumulative support for terms query
Fix test bug in query_string filter
[ TEST ] Add afterDistribution to testCorruptTranslogFiles
Tests : Fix randomizedtest fail
[ TEST ] Check accounting breaker is equal to segment stats rather than 0
Aggregations : Add cumulative sum aggregation
_all : fix RecoveryTarget creation .
Add support for headers . get ( ) ( )
Merge pull request from ywelsch / fix / block - delete - on - snapshot
Fix stream stats for DataStreamRestIT
update documentation with release 2 . 2 . 1
[ TEST ] Add SuppressSysoutChecks to DistributorDirectoryTest
Merge pull request from rmuir / rat
[ DOCS ] Describe ` indices ` token filter ( elastic / x - pack - elasticsearch )
[ TEST ] add support for binary type in aggregations ( )
[ ML ] Remove intervening # staleTask ( ) ( elastic / x - pack - elasticsearch )
Enable gradle gradle version
when using an index thread not alias in the mapping
Add index . asciidoc
Logging : Add', "'", 's', "'", 'option to the tests .
release 1 . 2 . 0
Update to AWS SDK 1 . 7 . 13
Added missing imports
Fix TimeSeriesLifecycleActionsIT . parsing ( )
Merge pull request from s1monw / renew_syn_on_merge
SQL : add catalog - type to the first parent ( elastic / x - pack - elasticsearch )
Marvel : Sort to find a buffer of failed shard allocation
Create Snapshots . Request ( elastic / x - pack - elasticsearch )
Add Intellij support for ingest processor ( )
[ DOCS ] Streamline title for search template API ( )
Make DeleteStep retryable ( )
Add favicon
clean code
Optimize customizing connections timeout ( )
Marvel : Add trap capabilities to whitelist tests
[ DOCS ] Add PGSync titles
[ ML ] Do not skip rows with missing values for regression ( )
remove redundant printStackTrace
SQL : Add new constraints functions to SYS TABLES ( elastic / x - pack - elasticsearch )
Docs : Note docs on the profiler
Remove IndexRequestBuilder . setType ( ) ( )
Check for the new ( )
Merge pull request from talevy / ingest - fix - grok - nocaptures
[ DOCS ] Add PGSync as community - supported integration ( ) ( )
Improve stability of UpdateIT
[ DOCS ] Fix typos in jump link
Merge pull request elastic / elasticsearch from rjernst / license_state_removal
SQL : Scripting runtime', implementation '( )
Merge pull request elastic / elasticsearch from rjernst / search_parser
[ DOCS ] Split Shadow - space . asciidoc ( )
[ ML ] Mute anomaly detection API ( )
[ DOCS ] Add . NET tag to beta API docs ( )
[ TEST ] add a lot of forgotten try with resources to wrap ElasticsearchResponses
painless : Add trap tests
Merge pull request from jdconrad / init2
[ DOCS ] Updating jvm . asciidoc
Disable testing conventions for idp - fixture ( )
[ ML ] Fix streaming the process cache request ( )
[ DOCS ] Fix typos in documentation
[ ML ] Remove record_count from ML mappings ( elastic / x - pack - elasticsearch )
test : fix logging in log log level
Added version 6 . 3 . 3 with bwc indices
[ ML ] Include message for DFA data frame analytics ( )
Alerting : Remove soft - deletes support in the unified highlighter .
externalize validate API to use the new API
[ DOCS ] Fix grammar fixes
Build : Make SchedulerEngine Discovery Discovery ( )
Merge pull request from cbuescher / introduce - shapebuilders
Merge pull request from rjernst / integ_test_deps
[ TEST ] Fix compilation
HLRC : Add Delete License API ( )
[ TEST ] Remove special field
[ TEST ] Fixed delete / 50_refresh REST test to work against multiple nodes
[ TEST ] Added basic slightly for indices . stats .
remove redundant logging
Reduce finalization of fixLag ( )
[ CCR ] Add create_follow_index privilege to test ( )
Merge remote - tracking branch', "'", 'upstream / master', "'
[ Kerberos ] Remove builders from RollupJobConfig ( )
[ DOCS ] Warn about query searching ( )
Revert " Revert " Ensure', Merge ')
 closes .
move to register a default settings
[ DOCS ] Explain ML datafeed details ( elastic / x - pack - elasticsearch )
[ DOCS ] Reformat ` remove_duplicates ` token filter ( )
rename client - groovy to lang - groovy
ILM should not fail with follower with remote index ( )
Merge pull request from jasontedor / immutable - be - gone
[ TEST ] Move from IntervalSchedule to CronSchedule .
[ Docs ] Update length - tokenfilter . asciidoc ( )
fix compile errors
[ DOCS ] Correct X - Pack tag usage API ( )
[ DOCS ] add missing comma
upgrade to amazon 0 . 3 . 0
remove redundant modifier
Include supported doc for doc snippets ( )
update headers
Fix unused test version in geo_polygon_query ( )
Merge pull request from jpountz / fix / compressedstring_equals
Merge pull request from rboulton / patch - 1
[ DOCS ] Fix typo in RoutingNode ( )
Merge pull request from polyfractal / bugfix / movavg_double_predict
remove DeleteByQuery code
[ DOCS ] Revise GS intro & remove redundant conceptual info . Closes ( )
[ Docs ] Update API to 1 . 0 . 0 - alpha1
Added dots in percolate case
[ TEST ] Actually perform the intended close . . .
[ TEST ] remove too abstraction
support from CamelCase when merging
Merge pull request from ywelsch / fix / handling_index_state
Merge pull request from elastic / revert - 18211 - docs / add_autosense_to_query_dsl
Merge pull request from rmuir / reindex_wait_for_completion
Merge pull request from jasontedor / remove - CONF_FILE
Merge remote - tracking branch', "'", 'origin / master', "'
[ DOCS ] Fix typo in mapping type docs ( )
[ Docs ] Added missing article
[ DOCS ] Reformat ` keyword_marker ` token filter ( )
Add compile errors to start content
[ TEST ] Add missing parts of auto - join params in tests ( )
bulk API : restore index . mapping setting
Docs : Fixed more cross doc link
[ TEST ] Add docs for integration tests
[ TEST ] Add additional logging to testCorruptTranslogFiles
[ DOCS ] Fix typos in context suggest
Remove SearchRequestParsers ( )
Add fromXContent method to HighlightBuilder
EQL : Fix issue with support for results ( )
when providing', this method 'to index name , closes .
lucene 4 : Upgraded ContentIndexSearcher
[ DOCS ] Update docs for the Go docs ( )
nicer logging
Fix Dockerfile instructions ( )
Merge pull request from javanna / enhancement / client_ingest_methods
Remove DiscoveryNode # id in favour of existing DiscoveryNode # getId
[ TEST ] Add await accidetially removed tests
Add wabisabi to Scala clients .
Merge pull request from rmuir / harden_integration_tests
Merge pull request from jpountz / fix / transport_index_duplication
[ TEST ] Change packaging', "'", 's LatLonShape ( )
[ TEST ] decreased number of replicas to 0 in refresh tests
Use versionless permission to read it ( )
Raised IOException on deleteBlob ( )
Merge pull request from biolds / master
Tip in detecting docs
Docs : Made cross - cluster snapshot experimental
[ DOCS ] Reformat ` keyword_marker ` token filter ( )
Add toString in FieldSortBuilder
[ TEST ] Use a clientSSLContext in LdapTestCase
Remove deprecated constructs calls with InternalClient
[ DOCS ] Add redirect for heap . asciidoc
Fix precommit
Query DSL : Allow for CamelCase field , closes .
Fix log
Fix testAutoFollowing aggregation on lambda aggregation ( )
[ TEST ] add a lot in forgotten test in InternalEngineIntegrationTest
Merge pull request from nik9000 / cleanup_search_request
Don', "'", 't swallow exceptions in Store # close ( ) . ( )
Extract hasValue
when not set to recover by default .
Merge pull request from rmuir / crazy_ubuntu
Data stream : add debug logging to ensure REST client tests ( elastic / x - pack - elasticsearch )
Minor cleanups in SyncedFlushService
Merge pull request from rjernst / client_jar_id

Aggregations : Adds cumulative sum aggregation
[ ML ] Skip MlMetadata in DataCountsReporter ( elastic / x - pack - elasticsearch )
Spelling : add put user by query ( )
upgrade to lucene 8 . 0 . 0 - snapshot - 08b8d116f8f ( )
[ TEST ] Fixed indentation in wait_for_active_shards test to allow
[ TEST ] Fix indexed test for greaterThanOrEqual fix
Do not reuse soft - deletes in terms ( )
Fix InternalSearchHit # hasSource to return the value upfront ( )
Merge remote - tracking branch', "'", 'dakrone / remove - disable - decider', "'
Merge pull request from rjernst / buildsrc_ide
Docs : Fixed more callout after the query_string query type
Rename ` options ` to top API ( )
[ ML ] Use JobTaskStatus # staleTask ( . . . ) to wait for job and datafeed status and
[ TEST ] Revert accidential massive iterations
Build : Switch to fedora - 25 ( )
Test : make sure that the test start is smallish
Add support for 7 . 0 . 0 - alpha2
Merge pull request elastic / elasticsearch from rjernst / license_state_removal
better exception message when failing to have a master node that is not removed
Reenable BWC tests after backport of ( )
Merge pull request from rjernst / remove / smart - type - wrapper
[ ML ] Improve detection of memory data frame analytics ( )
Fix Race in testGetSnapshotsRequest ( )
Merge pull request elastic / elasticsearch from martijnvg / deprecate
Merge pull request from talevy / more_details_factory_factory
For now do not guard ( )
Fixed update doc api
[ TEST ] Add additional logging to testCorruptTranslogFiles
Fix improved plugin issue in ObjectParser
Merge pull request elastic / elasticsearch from rjernst / deguice1
Update Put Status # Calendar to query ( )
Fix typo
allow to control the load cluster state . util setting
Docs : Add support for', the '/ _cat / nodes / fielddata APIs ( )
Merge pull request from jpountz / fix / transport_index_duplication
Watcher : Return useful fields with dots in the watch status ( elastic / elasticsearch )
Upgrade to lucene - 7 . 0 . 0 - snapshot - a0aef2f
Watcher : Hide support for STANDARD query ( elastic / x - pack - elasticsearch )
Merge pull request from rmuir / painless_compile_exceptions
[ DOCS ] Fine - tunes training_percent definition . ( )
[ DOCS ] Reformat ` kstem ` token filter ( )
[ TEST ] create the comparison
[ reindex ] Add support for regression in Ruby request documentation
Merge remote - tracking branch', "'", 'origin / master', "'
Add more tests for RemoteClusterService
Adjust BWC version for bad indices api ( )
Merge pull request from s1monw / cleanup_search_request
painless : Add more simple test for whitelist to Lists .
Merge remote - tracking branch', "'", 'es / master', "'", 'into feature / ingest
[ TEST ] remove leftover comment in test
[ TEST ] Mute IndicesCustomDataPathTests
Fix 8 . 0 . 0 - alpha2
Added unit tests for terms aggregation on more than one agg .
Make log4j dependency work
Multi / child : Allow to specify the ` _routing ` type , closes .
Remove Gradle version checks from checkPart1 and checkPart2 ( )
Merge pull request from ywelsch / fix / update - cs - with - routingresult
SloppyMath
fix compile error
Remove Redundandant modifier
Build : Add Bloodhound
allow to set the buffer size results in the ` node
Make parsing of transport logic , otherwise contains circular references
Merge pull request from javanna / enhancement / migrate_node_settings
Add classifier to tar . gz in docker compose ( )
[ DOCS ] Remove ` node . max_buckets ` setting ( )
[ GEO ] fix NPE in CompletionSuggesterBuilderTests
Merge pull request from rjernst / save_eclipse
upgrade to lucene 3 . 0 . 0
Merge pull request from rmuir / harden_integration_tests
Remove unused code
Update to AWS SDK 1 . 7 . 13
better handling of date facet
Allow BWC tests after backport of ( )
Aligned the Don', "'", 't bulk requests .
[ DOCS ] Mention Integer . MAX_VALUE limit for http . max_content_length
Implement ctx . asciidoc ( )
SQL : Fix incorrect', "'", 's profile code
SQL : Add catalog flag to sequence mechanism ( )
expose comparator to search requests , closes .
[ DOCS ] Update migration
Mute InternalAutoDateHistogramTests # testReduceRandom ( )
Add breaking changes to migration docs
[ TEST ] Fix incorrect assert to wait for yellow to wait for green
[ Security ] Check CryptoService for null fields ( elastic / elasticsearch )
Disable BuildExamplePluginsIT
[ CORE ] Remove special doc setting
[ DOCS ] Merges duplicate pages for Active Directory realms ( )
[ DOCS ] Add redirect for ` search . max_buckets ` ( )
Balance old shards in TransportNodesListGatewayStartedShards files
Merge pull request from jpountz / remove / multi_field
lucene 4 : Upgraded ContentIndexSearcher
Watcher : Move YAML rest tests for PKI to core ( )
[ TEST ] [ TEST ] [ test ] Kill for java / set ( )
 )
[ DOCS ] Add overview sections to the and changes in the profiler
Fix yamlBuilder ( ) ( )
Remove the typo in the docs . ( )
lucene 4 : Upgraded ContentIndexSearcher
update headers
Merge pull request elastic / elasticsearch from PhaedrusTheGreek / group_search_noattrs
[ Transform ] Fix support for more test for more data frame analytics ( )
[ DOCS ] Add missing default value to cat . json
Migrate BWC tests for clone index API ( )
[ DOCS ] Update link to 2018 ( )
[ TEST ] Add additional debug output to HasChildQueryBuilderTests
Upgrade to lucene - 8 . 0 . 0 - snapshot - b313618cc1d ( )
[ ML ] Better detection of binary input in find_file_structure ( )
[ DOCS ] EQL : Move comparison operator defs
Mute TransportClientNodesServiceTests # testListenerFailures
Test : finish malformed tests
[ DOCS ] Format the addresses of a started with the profiler
[ TEST ] Fixed version of translog type to allow
rename the plugin transport intro to use it specific service
Merge pull request from elastic / master
Watcher : index index type on field type ( elastic / x - pack - elasticsearch )
Add integration tests
Add missing {
Fix a typo in validate lookup api
Scripting : Add simple tribe type to whitelist tests
bump out of 2 . 2 . 0
Update ( )
Packaging : Add aws - plugin to elasticsearch - plugin
Installing maps processing
[ Monitoring ] Remove Dedicated IndicesStatsCollector ( elastic / x - pack - elasticsearch )
Remove ClusterService . Type ( )
Limit request size on transport level
Added 2 . 3 . 9 - SNAPSHOT and 1 . 4 . 4 - SNAPSHOT versions
Fix PrimaryAllocationIT # testForceStaleReplicaToBePromotedToPrimary ( )
[ TEST ] failure accounting is null in the IT
The node and stats
[ DOCS ] Fix typos in ` terms ` aggregation ( )
[ TEST ] Add unit test tests .
SQL : cover the use of date field parsing ( elastic / x - pack - elasticsearch )
[ ML ] Mute ml MlDistributedFailureIT test ( )
Merge pull request from nik9000 / plugin_docs
Add async_search trace Retry to HLRC ( )
Remove unused modifier .
Fix integ tests for windows ( )
Docs : Adding Hebrew analyzer
[ Zen2 ] Change . gitignore ( )
Merge pull request from s1monw / serialize_interrupted_exceptions
remove references to tribe test in security tests
Adjust BWC version after backport ( )
 )
Use versionless version bwc in packer usage ( )
[ ML ] Wait for job to close in test ( elastic / x - pack - elasticsearch )
Add some method to ESLoggerUsageTests
add debug logging to start row files
fix test
HLRC : Add spi client to a HLRC "
Handle lastGc
[ DOCS ] Add missing icons to transform HLRC APIs ( )
rename field to be sorting in _all aggregation ( ) ( )
Use a comment in the docs ( )
Merge pull request from jpountz / fix / transport_index_duplication
[ TEST ] add more debugging to testCorruptTranslogFiles
Mute TransportClientNodesServiceTests # testListenerFailures
Remove ` CcrRepository ` CCR ( )
nicer code
[ TEST ] Fix test bug in MultiOrdinalsTests ( )
Remove some generics from SearchContext .
Add 0 . 19 . 5
Merge pull request from rjernst / fix / 10659
Send non - null searchSourceBuilder in searchRequest for now
[ TEST ] Add security . teardown to security tests
Fix ( )
[ TEST ] Fix for for parent / length of parent / child field .
Mapper : Allow to set automatic fields by ` thread_per_shard ` , closes .
Remove PreProcessModule
Plugins : Remove dead code from ScriptContext ( )
Record Memory comments
Merge pull request from cbuescher / feature / query - refactoring - spanmultiterm
remove old leniency
Adjust retention lease background Memory ( )
Aggregations : Add support for Ctrl - aggregation . properties ( ) .
Fix NPE in NodeConnectionsServiceTests ( )
[ DOCS ] Remove ` type ` in disk stats docs
Docs : Use the new experimental annotation .
Improve search stats for shard fails ( )
Remove DiscoveryNode # setTemplate ( )
Merge pull request elastic / elasticsearch from rjernst / deguice1
Add setup to the test for the docs .
Watcher : Properly index state action ( )
Docs : CONSOLEify plugin . asciidoc
Mute FullClusterRestartSettingsUpgradeIT
Add missing SystemD support ( )
Internal : Use corruption strikes of allocation execution
Mute the test more in tests
fixed constants
Inline NettyHttpChannel # getStatus
Added settings filtering
TEST : Mute FullClusterRestartIT ( ) ( )
Docs : Fix syntax
[ DOCS ] Fix typo in _analyze query docs ( )
ingest : Fix randomizedtest fail
Limit realm and ldap in an index ( elastic / elasticsearch )
Scripting : Introduce support for casting ( )
Restore NoSuchDirectoryException on deleteBlob ( )
Merge pull request from rmuir / xlint_plugins
Fix Infinite docker test ( )
Dates : More strict parsing of ISO dates
Build : Fix randomizedtest Processor ( )
Fix Infinite Loops in ObjectParser
Merge pull request from jdconrad / cast
Mute TimeSeriesLifecycleActionsIT . testRolloverAlreadyExists ( )
Remove type in ` terms ` mapping ( )
upgrade to amazon 0 . 7
[ DOCS ] EQL : Add example sensitivity for EQL search , docs
[ DOCS ] Add PIT to multiplexer in documentation
Update x - pack to 1 . 3 . 0 / Beta1
Added the API definition for the rest client spec ( )
[ TEST ] add additional logging to testCorruptTranslogFiles
Text fix should use the same name ( )
TESTS : Fix Assertion About Deleting Shard Dir ( )
[ DOCS ] Add supported ESS support
[ DOCS ] Remove ` twitter ` dataset in threadpool docs ( )
Revert " Ensure domain_name setting for AD realm is present ( ) "
[ DOCS ] Reformat ` remove_duplicates ` token filter ( )
HLRC : Add support for put request
Merge pull request elastic / elasticsearch from jasontedor / sets - be - gone
Merge pull request from gmarz / feature / virtuallock
[ DOCS ] Add missing widget for bucket_span field locations ( )
Added 2 . 3 . 9 - SNAPSHOT and 1 . 4 . 4 - SNAPSHOT versions
[ TEST ] Mute CcrRetentionLeaseIT . testRetentionLeaseIsRenewedDuringRecovery ( )
move to set remaining fields field
allow to control the default value of the ` search ` aggregator .
Mute TransformIntegrationTests # testSearchTransform
Merge pull request elastic / elasticsearch from jdconrad / deprecate
sync , closes .
Persistent Tasks : Remove retries setting ( )
Fix indentation
Remove ( )
[ DOCS ] Update ` kstem ` token filter ( )
Query refactoring : Add ` reverse_cout ` and ` reverse_term ` to ` order ` options .
Docs : Fixed link to percolate_format description
Bootstrap check for OnError and seccomp
[ ML ] Fix BWC versions in data frame analytics ( )
Fix msearch Retry test ( )
Validate ` _type ` updateable to POST ( )
Consolidate . gitignore entires for eclipse
[ DOCS ] Clarify title for Beats security ( elastic / x - pack - elasticsearch )
[ TEST ] Add SuppressSysoutChecks to DistributorDirectoryTest
Alerting : Split alerting into packages
Extend duration of fixLag ( ) ( )
[ TEST ] Fix indexed setup
[ ML ] Remove Dedicated IndicesStatsCollector ( elastic / x - pack - elasticsearch )
[ TEST ] Add back accidetially removed test logging
painless : Add trap support to the type ( )
[ TEST ] Remove docs for pending cluster update update
Replace Painless . version ( )
Merge pull request from s1monw / renew_syn_on_merge
[ TEST ] mute REST test tests that uses it
Adding a debug query
Upgrade to Netty 4 . 6 . 0
Remove ParseFieldMatcher usages from RestSearchTemplateAction
Text Query : Add Terms API , closes .
lucene4 : unit tests cleanup
Extract RoutingNodes and PendingClusterTasksResponse ( )
Docs : Documented Shield pages ( )
Merge pull request from rjernst / deprecate_exception_hurter
Update to master 4 . 10 . 2
Internal : Remove unused API
Add licensing http test
Merge pull request from rjernst / fix / 12329
Internal : Remove dead code from delete mapping
Remove Painless test failure in tests
MINOR : Fix Assertion About Deleting Shard Dir ( )
Changed checks for _all parameter in a single manner
Add wabisabi ( )
prepare release elasticsearch - cloud - aws - 1 . 14 . 0
removed infinite test in shield
update to 0 . 19 . 0
Fix temporary Retry timeout ( )
Disable roles metadata when data ( )
Fix name of hidden / include_type_name calls in REST tests ( )
Do not resolve all chars
[ TEST ] Fix SimpleQueryTests . testScript_ExplicitSingleValued_WithParams
Build : Reorganize src roots
Send explicit settings when iterating shard fails
Build : Fix vagrant tests for xplugins
[ ML ] Improve assertion with prediction field data ,
[ TEST ] Improve rest endpoint for missing . json
[ CORE ] Move as much as possible into the abstract Engine
improve spelling
[ TEST ] Use a dedicated TestLogging for recovery
[ TEST ] Add security . security . node to be set
HttpInfo to UserError on circuit than Streamable
Add missing {
only ask for the QueryShardContext stats
SQL : Fix display size of DATE functions ( )
Verify all shards if a transports are bound to
Remove ParseFieldMatcher usages from RestSearchTemplateAction
less Query when shutting ( )
support rebalance when recovering to read ( deleted ) open file handles , closes .
Remove unused PR from RestSearchTemplateAction
EQL : Remove unused constant equality ( )
[ DOCS ] Add xref to multiplexer token filter docs ( )
Merge pull request from ywelsch / fix / compressedstring_equals
Merge pull request elastic / elasticsearch from rmuir / simplify_ssl_test
Add favicon
[ DOCS ] Update plugin . asciidoc
[ DOCS ] Add docs for ` geo_match ` token filter ( )
[ DOCS ] Fix typo in jump link . ( )
Update Settings comments
Remove the LocalTransport of the config
Transport Client : fixed ` reverse_cout ` and ` reverse_term ` , closes .
Remove ParseFieldMatcher usages from RestSearchTemplateAction
Add _cat / replicas
Rename common', "'", 's into backport " ( )
Include', "'", 'local', "'", 'version in watcher email ( )
Added transport license to action description
Internal : Fix compilation issue with latest users and added security
Update to elasticsearch 1 . 0 . 0 . Beta1
Added bwc tests on elasticsearch - parent
Merge pull request from ywelsch / fix /', refact
Merge pull request from rmuir / crazy_ubuntu
[ DOCS ] Add tab widget for default log locations ( )
Disallow a new version to be callback for ` disk . disk ( )
Added missing getters to FsInfo . IoStats class
inherit from BroadcastOperationRequest
SQL : Make Literal for NamedExpression ( )
Update to AWS SDK 1 . 7 . 13
Introduced an static file thread pool
Adjust enables interface blacklist ( )
Do not use AssertionError ( )
Watcher : Ensure triggered watch is deleted ( )
Use the response', "'", 's scripting / aggregations in the head ( )
[ DOCS ] Streamline GS anchors for GS custom getting started page
[ TEST ] Remove double request based on search response
[ Security ] Handle leader privilege with empty transform ( )
Update to Lucene 4 . 6 . 1 / Elasticsearch 1 . 0 . 0
Enable IntelliJ import instructions
[ TEST ] Painless Painless comparison
Shard Allocation : Add ` reverse_cout ` and ` reverse_term ` to ` order ` options , closes .
Mute FullClusterRestartSettingsUpgradeIT
TransportInfo to implement Writeable rather than Streamable
[ DOCS ] Update data - index to v2 . asciidoc ( )
- TokenFilters and Tokenizers are loaded via the original ( more maintainable ) way as was in the previous commit .
[ TEST ] Mute IndicesCustomDataPathTests
Switch SearchAfterBuilder to writeGenericValue
Stop atomic as GC
Add more asciidoc for index state normalizer ( )
[ TEST ] fix CommonTermsQueryBuilderTest
add the 1 . 5 . 0 version
HLRC : Add pipeline type to per search ( )
[ Docs ] Add ` remote - data - ` ` ( )
[ DOCS ] Call - configuration to Elasticsearch Ref ( )
[ DOCS ] Fix typos
Merge pull request from rmuir / shave_mustache
Marvel : Add MonitoringClientProxy
Remove PreProcessModule
Add docs for the new field name ( )
remove the settings_require_units , when it is not needed
Watcher : make SchedulerEngine job handling threadsafe ( elastic / elasticsearch )
Fix SearchableSnapshotActionIT ( )
Adapt input stats for match ( )
improve code
refactor InternalClusterInfoService
minor chagne
Make job status of bucket and
Docs : Remove use of keyword_marker on the lang analyzers
remove cruft
[ Docs ] Remove docs for field names in the docs .
[ DOCS ] Fix typos in RoutingNode ( )
Query refactoring : query_string
Query DSL : ` constant_score ` and ` reverse_term ` to ` order ` as ignored in the query .
[ DOCS ] Add tab widget for examples . ( )
[ DOCS ] Fix typos in percolate_format field message ( )
add transport service
[ ML ] Fix online updates with custom rules referencing data ( )
review comments
[ DOCS ] Adds abbreviated titles
[ TEST ] Add SuppressSysoutChecks to DistributorDirectoryTest
Merge pull request from rjernst / ingest_plugin_api
Merge pull request from dongjoon / hll - be - gone
[ DOCS ] Reformat the grok search thread pools ( )
[ DOCS ] Describe shared APIs to use two value API ( elastic / x - pack - elasticsearch )
Cleanup agg conflicts
Fix test some long test
[ DOCS ] Add aggregation . max_buckets . ( )
Enable CCR testing for idp - fixture ( )
Add faker
[ TEST ] Add static test to testCorruptTranslogFiles
Update ` IndexShard # refreshMetric ` via a ` ReferenceManager . RefreshListener ` ( )
Plugins : Fix IndexRecoveryCollectorTests
Removed unused code
[ DOCS ] Added elasticsearch_nodes to Elasticsearch blacklist
apply review comments
[ DOCS ] Update monitoring to use two page
inherit from BroadcastOperationRequest
Add missing comparator implementation to plugin ( )
Fix ingest pipeline aggregation merging ( )
optimize MockStore to read writeables
only fail if index is not in indices . Issue
Mute use 20_minimal_body
Fix logging
[ DOCS ] Correct enrich index files ( )
Fix indentation in geospatial phrases ( )
[ DOCS ] Fix GET default in find_file_structure query docs ( )
add also level
Merge pull request from rjernst / unused
Introduced ` node . version . single_type ` to ` ctx ` as the default settings .
Add support to the main API docs
Update gradle enterprise plugin to latest version ( )
SQL : cover LEFT and RIGHT in DATEDIFF ( )
[ TESTS ] Licenses can only be present
Improve merge error
Basic request - level recovery
Dates : More strict parsing of ISO dates
[ TEST ] Only use the clients from the node .
Fix test to use watcher relocation tests
[ rest : Remove dead & http type .
[ TEST ] packaging : packaging tests for packaging tests
Merge pull request from biolds / master
Add Failure to CCR ( )
[ TEST ] Add IndexShardOperationPermits to DistributorDirectoryTest
Add support for STANDARD license
[ TEST ] Add SuppressSysoutChecks to DistributorDirectoryTest
Move owner to elastic
change rollover rest - api placeholder name
[ DOCS ] Chunk link to community upgrade docs ( )
painless : Adds synthetic boost
Mute RecoverySourceHandlerTests . testOpenContexts
Remove SecurityTransportModule
Change v3 . 0 . 0
Merge pull request from nik9000 / painless_generated_skip_checkstyle
Mute failing SamlAuthenticationIT test ( )
Merge pull request from jpountz / fix / 12329
add a whitespace to simplify pages file
Mute SearchStatsIT . testOpenContexts
Improve BWC issue after backport ( )
improve code
Tribe simplify buffer requests , closes .
Remove SearchPhase interface ( )
[ TEST ] Mute enrich mapping REST test ( )
[ ML ] Rename datafeed indexes to indices ( elastic / x - pack - elasticsearch )
apply review comments
Disable reindex Manager
[ DOCS ] Remove beta label from normalizers ( )
Added helper methods test
SQL : Make SqlAction methods ( )
[ DOCS ] Add PKI realm configuration details ( )
Enable tests for now
Painless : Fix documentation for painless painless ( )
Remove buildResources from BuildPlugin ( )
[ DOCS ] Added shared configuration client ( )
Build : Fix license issue for integ tests ( )
[ Transform ] Fix RepositoryURLClientYamlTestSuiteIT
upgrade to the max size
Enable gradle build for JDK 10 gradle ( )
Add support for auto_generate_synonyms_phrase_query ( )
update for yaml the Rest tests ( )
Merge pull request from nik9000 / varargs
Merge pull request from cbuescher / feature / query - refactoring - constantscore
Make FsBlobContainer Listing Resilient to Concurrent Modifications ( )
test : fix all test test
Test : Now don', "'", 't create ScoreAccess properly
SQL : Fix incorrect test of SqlResponse ( elastic / x - pack - elasticsearch )
[ ML ] Remove encapsulated BWC tests
[ TEST ] Add missing license
Merge pull request from rjernst / deprecate_exception_hurter
Internal : Simplify SecurityContext dependencies
Fix FVH ( )
Remove old - license
Merge pull request from rjernst / client_jar_id
[ ML ] Update DataStreamDiagnostics to JobResultsProvider ( elastic / x - pack - elasticsearch )
Clean up IpRangeAggregation to fix ( )
[ DOCS ] Add PIT to search book docs ( )
doc
Fix test bug
Addressed in plugins tests
Rename syskeygen
[ TEST ] only watcher security plugin
Fix BWC tests after backport of ( )
Merge pull request from jpountz / enhancement / stricter_metadata_parsing
Merge pull request from rmuir / more_indy_typing
Support client of client requests ( )
remove cruft
Merge pull request from jimferenczi / copy_to_dynamic_object
Better exception message while trying to extend the shard state
[ doc ] Correct typo in _analyze docs ( )
[ DOCS ] Linked cat - level . asciidoc ( )
[ ML ] [ Inference ] changing model model plugin ( )
Fix CoordinatorTests ( )
clean code
- TokenFilters and Tokenizers that are loaded or non - index operation .
Merge pull request from jdconrad / xlint2
Merge pull request from nik9000 / java_8_for_package_tests
Fix Infinite Loops ( ) ( )
Merge pull request from jasontedor / heap - size
[ DOCS ] Add reference to Beats security API ( )
Fix length of FsInfo # dataNode . ( )
Tests : Change vagrant tests in packaging tests
Recovery : add debug logging .
Cleanup : Remove generics in XContentBuilders
[ DOCS ] Fix cannot must typo . ( )
Clean up access ( )
when compression of match_all field
Build : Change CName in Http Publish Addr ( )
bump master to 2 . 0 . 0 . beta1 - SNAPSHOT and fix broken assert .
[ TEST ] Pass perform the comparison
Merge pull request from rjernst / save_eclipse
[ DOCS ] Fix hardcoded link to ES ref ( elastic / x - pack - elasticsearch )
[ TEST ] Don', "'", 't use the MlRestTestStateCleaner class for the rest client ( )
Fix opening docs ( )
remove the skip version in BWC tests
Merge pull request from rjernst / client_jar_id
always the serialization builder ( fail ) with the same logging
[ ML ] Remove record_count from the new mappings to stop process ( elastic / elasticsearch )
[ TEST ] Also sure nodeSettings and wait for termination
Fix DateTime - tools of 39525 .
reenable bwc test now that is merged ( )
Cross - cluster search in write docs ( )
Fix NPE due to MaxAggregatorTests ( )
Update getting - started . asciidoc ( )
fix NPE bug in PropertyPlaceholder .
Create branches according to elasticsearch versions
Move', ['SCRIPTING tests to new MapperTestCase ( )
Adapt to elasticsearch 1 . 0 . 0
Fix GCS plugin tests
[ TEST ] Mute IndicesCustomDataPathTests
: fix for test ( )
elasticsearch . R is abandoned ( )
Merge pull request from nik9000 / license_check_for_dirs
Merge remote - tracking branch', "'", 'upstream / master', "'", 'into feature / sql_2
lucene 4 : Converted ContentIndexSearcher
Fix tests for now
lucene 4 : Moved from FieldSelectors to FieldVisitors . Removed BaseFieldVisitor # reset and changed SourceFieldVisitor and UidFieldVisitor to singleton to prototype .
Add xContent shuffling to aggregation tests
Updated sorting docs to 5 . 0 . 0 - alpha4
Add for / _cat / fielddata
Update S3Gateway to be deterministic
Merge pull request from s1monw / cleanup_search_request
Fix Incorrect Requeset in Snapshots ( )
Upgrade to Netty 3 . 2 . 1
remove error message ( )
Adjust BWC version after backport unsupported ( )
Depend on elasticsearch - parent
[ DOCS ] Add ` recovery ` token filter docs ( )
[ ML ] Rename datafeed indexes to indices ( elastic / x - pack - elasticsearch )
Added the API definition for ` _cache `
ProfileScorer should propagate ` setMinCompetitiveScore ` . ( )
remove timer service
[ DOCS ] Adds supported datafeed API docs ( )
Mute WatcherRestIT rest test methods ( )
[ DOCS ] Added explicit " lang " field to documentation of script score definition
Test : Increase node and stop all shards in the mapping
Release : Fix randomizedtest fail
minor simplifications in SyncedFlushService
prepare release elasticsearch - analysis - stempel - 1 . 0 . 0
[ Monitoring ] Check CryptoService for bulk metadata ( )
[ Tests ] Fixed whitespace in search API test for now

