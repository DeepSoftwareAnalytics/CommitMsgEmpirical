Propogate rejected execution during bulk actions ( )
Do not catch throwable
[ TEST } unmutes SearchAsyncActionTests and adds debugging info
Alerting : Improve error handling in the RestInterface
Partial draft for Java Update - by - Query
Simplify searchable snapshot shard allocation ( )
latest assembly
Enforce realm name uniqueness ( )
Fixes MultiMatchQuery so that it doesn ' t provide a null context ( )
Cut over IPFieldMapper to parametrized form ( )
Improve deletion of corrupted snapshots
Add test for global checkpoint sync with security
Merge pull request from jasontedor / property - placeholder - null - value
lucene4 : add support for omit_norm setting to numeric types and don ' t omit norms if boost is not 1 . 0
[ TEST ] fix SearchIT assertion to also accept took set to 0
Added release notes for 5 . 0 . 0 - alpha3
Add points to SegmentStats .
Add lenient option to match & multi_match queries .
Update to Lucene 4 . 10 . 2
[ TEST ] Remove explicit network mode - not needed here
Test : RelocationTests . testMoveShardsWhileRelocation should wait on nodes to start before asking for a client
log if sigar was loaded as well
Preserve existing mappings on batch mapping updates
Tests : Add unit tests for JarHell
Convert several direct uses of Streamable to Writeable ( )
Add an option to force _optimize operations .
Document that the PUT mapping API with the _default_ type overwrites instead of merging
Merge pull request from rmuir / groovy_factor_out
Docs : Documentation for the upcoming SQL support of frozen indices ( )
upgrade to jackson 2 . 1 . 1
Make peer recovery send file chunks async ( )
S3 Repository : Add back repository level credentials ( )
Fix test bug in CacheTests # testDependentKeyDeadlock
Query DSL : queryString - allow to run against multiple fields , closes .
Use IOUtils # close ( ) where needed
Introduce elasticsearch - env for Windows
test : added test for http input querying api other than _search
Build : Allow customizing wait condition and cluster settings
expose node in NodeModule and Environment in Node
change setting name from index . warm . enabled to index . warmer . enabled
Build : Disable licenses checks for all xplugins
Support numeric bounds with decimal parts for long / integer / short / byte datatypes ( )
Merge pull request from s1monw / kill_es_deletion_policy
Cut bucket_script and bucket_selector to registerPipelineAggregation
fix compile errors after merge
Merge pull request from masaruh / same_shard_host_setting
Merge pull request elastic / elasticsearch from nik9000 / trask_status
Refactor create index service to be unit testable
Replaces usage of ` StringBuffer ` with ` StringBuilder ` and fixes stray
Remove unnecessary hash map copy in o . e . b . Security
Update the Flush API documentation ( )
Clarify remote clusters ' use of transport layer ( )
[ ILM ] rolling upgrade tests ( )
Fix another simulate example in ingest docs
Wire up GeoDistanceAggregation ( )
[ DOCS ] improve explanation of dynamic mapping setting ( )
Switch SqlGetIndicesAction after merge
Make painless ' s assertion about out of bound less brittle
Depend on elasticsearch - parent
second phase at optimizing merging / parsing large new mappings
moving debian package to maven
add the ability to run as another user
Catch EsRejectedExecException on cluster info reschedule
Remove debugging logic
Revert " Disable BWC tests for build issues "
[ DOCS ] EQL : Fix whitespace in ` stringContains ` docs
[ DOCS ] Adds link in datafeed indices_options ( )
LLREST : Introduce a strict mode ( )
Convert parent - join mappers to parametrized form ( )
Remove DummyTransportAddress ( elastic / elasticsearch )
Upgrade Gradle docker - compose plugin ( )
general indexlifecycle cleanup , rest tests , reintroduce integ - test ( )
Tidies up unit tests
[ DOCS ] Remove unused docinfo . xml files ( )
[ DOCS ] Document ` transport ` and ` http ` node stats ( )
Docs : fix health response test
Testclusters support for multi node clusters ( )
More read - only - allow - delete docs ( )
Use unwrapped cause to determine if node is closing ( )
inner hits : Don ' t fail if an object is specified as a nested value instead of an array .
[ ldap ] Migrate to using UnboundID SDK for LDAP
Watcher accounts constructed lazily ( )
Exclude specific transport actions from request size limit check
[ DOCS ] Added kibana_dashboard_only_user role ( elastic / x - pack - elasticsearch )
properly bind ClassSet extensions as singletons
http client : Don ' t append ' ? ' to the url if there are no query string options defined .
QL : Extend NodeSubclass to read classes from jars ( )
Core : DistributorDirectory shouldn ' t search for directory when reading existing file
Remove unused import from NodeInfo . groovy
[ DOCS ] Rewrite ' rewrite ' parameter docs ( )
Updated watch_history template to better support aggregations
Added percolate statistics to indices and node stats .
Build : If SHA files have changed , explain how to update them in the license check exception
[ ML ] Mute data frame analytics BWC tests
Test : fix test bug with incorrect date math expressions
Marvel integ test should stop and start exporters between tests . The ongoing activity messes with ESIntegTest level clean ups
Close xcontent parsers ( partial ) ( )
Add elasticsearch - nio jar for base nio classes ( )
Added comment to explain MarginFragListBuilders relation to SimpleFragListBuilder
Removed logger prefixes when using @ TestLogging annotation
TOKEN_SERVICE_ENABLED_SETTING enabled if HTTP_SSL_ENABLED ( elastic / x - pack - elasticsearch )
Remove unnecessary throws IOException in CompressedXContent . string ( ) ( )
Added _shield to the verification step . Closes elastic / elasticsearch .
EQL : Remove list classes from EqlSearchResponse ( )
Add List Append Utility Method ( )
make description optional as part of a Pipeline
[ ML ] refactor tasks classes and add new abstract executor ( )
Mute failing AsyncTwoPhaseIndexerTests
Add tests for cross cluster ` _field_caps ` API ( elastic / x - pack - elasticsearch )
Tests : Control reallocation in upgrade test to stop rebalancing causing upgrade requests to be lost .
Completely remove Preconditions . checkArgument
[ TEST ] NettyTransportMultiPortIntegrationTests used wrong random to
rename ElasticsearchResponse # getFirstHeader to getHeader
[ ML ] Adjust versions and unmute BWC tests after backporting ( )
Plugins : NPE when plugins dir is inaccessible
[ DOCS ] Copy watcher docs from x - pack to x - pack - elasticsearch repo ( elastic / x - pack - elasticsearch )
cut over indexing slow log
Check index under the store metadata lock ( )
Docs : Match the examples in the description ( )
[ ML ] Mute data frame analytics BWC tests
Increase shard inactive time to 1h in upgrade tests ( )
Fix bug in BucketMetrics path traversal ( )
Optimize version map for append - only indexing ( )
REST spec : Added the verbose flag to indices . segments
Expose dist / pre / post options for SpanNotQuery
[ ML ] [ Transforms ] allow executor to call start on started task ( )
Add audit messages for Data Frame Analytics ( )
Fix test on Linux ; setting name missing ' total . '
Convert ` cluster . name ` to the new settings infra
Ensure ESTestCase is initialized before we run tests
Merge pull request from rmuir / geoip - nodns
add lang parameter to all script constructs , allowing for ( later ) custom script languages
Create a class to hold field capabilities for one index . ( )
Build : Fix dependency licenses check to correctly skip projects without dependencies
Remove Favicon Special Path in RestController ( )
Refactors ScriptQueryBuilder and Parser
[ Docs ] Fix the numbering in the annotated_text example ( )
Renamed [ [ getting - started ] ] to [ [ getting - started - with - shield ] ] to avoid 301 redirect
force default codec
Marvel : Use versioned index templates
Adapt to changed method name
Increase gradle heap to 2048m
ACKS : Fix some issues when a transport client was used .
Persistent Tasks : check the current state in waitForPersistentTaskStatus ( )
Mark SearchSourceBuilder as ToXContentObject ( )
Templates : Add compileTemplate method to ScriptService for template consumers ( )
Minor fix to a failing REST test due to the prior check - in .
Index logstash_stats . pipelines . events . duration_in_millis ( elastic / x - pack - elasticsearch )
[ Discovery ] join master after first election
Tighten which classes can exit
[ Test ] Handle file permissions for Windows ( )
Correct the serialization version for index field caps .
Add @ AwaitsFix for
move DatabaseReaders initialization to IngestGeoIpPlugin # onModule
Fix compilation for only one script language extension
Remove unchecked warnings rendered unnecessary
Remove schedule field from EnrichPolicy ( )
Pass processConnectTimeout to the method that fetches C + + process ' PID ( )
Update Index usage to elasticsearchelastic / elasticsearch
EQL : Add ? character to : operator ( )
Improve testTransportStatsWithException test ( )
[ Rollup ] Re - factor Rollup Indexer into a generic indexer for re - usability ( )
SQL : Don ' t override calendar for Thai locale
tests : fix test bugs
Fix NumberFormatException in Simple Query String Query
Added readTimeout , writeTimeout , readAcknowledged and writeAcknowledged methods variation without version as parameter , to be used by api that always read and write those parameters . Helps avoiding to call the variation that accepts a Version with actual version null .
Changed LDAP searches to NOATTRS in order to avoid returning unecessary
Query Refactoring : Move MoreLikeThisQueryBuilder validation to constructor
test : increase timeout
Remove Guava as a dependency
[ TEST ] Adjust version skips for movavg / movfn tests
Add _cat / help ( halp ! )
Ingest : Enable Templated Fieldnames in Rename ( )
Fix key_as_string for date histogram and epoch_millis / epoch_second format
Moved waiting for scheduler started logic into StartSchedulerAction . TransportAction and moved the logic that was original there to a new action named InternalStartSchedulerAction .
document suggest stats being merged with search stats
Merge pull request from pjcard / patch - 1
Revert temporary compiler bug workaround
[ TEST ] Wait for primary allocations before restart
Rewrite to unbounded range query if relation to query is WITHIN
Register DFA jobs on putAnalytics rather than via a separate method ( )
Fix broken SQL security test
httpAddress - > http_address , closes .
Handle delete document level failures ( )
Calculate changed roles on roles . yml reload ( )
Watcher : Improve assertion output in flaky test
Build : Fix reproduce info for methods with ( or ) ( )
Move upgrade test to upgrade from version 2 . 3 . 3
Tests : Fix array out of bounds exception in TemplateUpgradeServiceIT
Update version of TaskInfo header serialization after backport
Remove testing of 2 . x indices in DecayFunctionScoreIT .
Fix bug when searching concrete and routing aliased indices
Replace usages of XPackPlugin with the LocalStateCompositeXPackPlugin ( )
PkiAuthenticationTests & SslIntegrationTests should properly handle ipv6 addresses
Mappings : Make lookup structures immutable .
fixup issues with 32 - bit jvm
Move IndicesAliasesRequest # concreteAliases to TransportIndicesAliasesAction ( )
Merge pull request elastic / elasticsearch from s1monw / fix_settings_filtering
wrap checked TimeoutException with runtime ElasticSearchTimeoutException
[ STORE ] Simplify reading / writing from and to BlobContainer
Serialize Monitoring Bulk Request Compressed ( )
Reindex : Fold " with all deps " project into reindex ( )
Build : Add pom building and associated files to rest api spec jar ( )
Updated doc versions to 7 . 0 . 0 - alpha1 unreleased
Remove with ( out ) - system - key tests ( )
* Update the alert version on each write
Add test that throws exceptions during search execution
upgrade to latest testng , improve console output when running test , add more options as env vars when using maven
Allow different source sets from forbiddenApis ( )
shield : add rest API spec for authenticate api
require hosts when creating RestClient . Builder
[ CCR ] Made leader cluster required in shard follow task .
Moved the stopping log info message
Periodically try to reassign unassigned persistent tasks ( )
move to 1 . 2 . 0 snap
Revert " Exploit DiscoveryNode immutability . . . "
fixed compile errors
Update to elasticsearch 1 . 0 . 0 . RC1
Merge pull request from brwe / fix - cicular - reference - exception
remove thread leak filter , should have been removed in
Emit settings deprecation logging at most once
Update to GCE API v1 - rev59 - 1 . 20 . 0
Move GapPolicy and resolveBucketValues ( ) to static helper methods
small refactor in index settings , allowing for future dynamic changes to it
Deprecate accepting malformed requests in stored script API ( )
Routing : When specify in the mapping _routing required , a delete without explicit routing value should automatically be broadcasted to all shards , closes .
Simplify Log4j shutdown hack test
Docs : Add documentation for the ` doc_values ` setting on the ` boolean ` field type .
Watcher : Only load active watches on load ( elastic / x - pack - elasticsearch )
improve javadocs
Remove BouncyCastle dependency from runtime ( )
upgrade to lucene 5 snapshot
[ TEST ] Fix ExistsMissingTests failures .
Mute testLookupSeqNoByIdInLucene
Fix ingest java week based year defaulting ( )
Fix testShouldPeriodicallyFlushAfterMerge ( )
Add a fineName field to index the attchment fileName
Handle existence of cgroup version 2 hierarchy
[ DOCS ] Remove cat request from Index Segments API requests ( )
[ TEST ] Remove current locale log line from ShieldRestTests
Check that we must have one and only one scope for a Setting
fix typo
XContentBuilder throws NPE on null Boolean , closes .
Fix long path issue of bwc checkouts on Windows ( elastic / x - pack - elasticsearch )
Rework output normalization in build - tools func tests ( )
Introduce 7 . 3 . 1 version ( )
Fix compilation after merging XContent changes
Trace log all shard failures if an exception is attached .
Revert " quiet flakey gce tests "
Add active directory bind user and user lookup support ( elastic / x - pack - elasticsearch )
[ DOCS ] Fix REST test execution line in testing documentation
Merge remote - tracking branch ' dakrone / add - noop - it '
[ SPEC ] The scroll_id is no longer a required qs param
Mute data stream YML tests until backport ( )
Avoid polluting download stats on builds ( )
Internal : made original indices optional for broadcast delete and delete by query shard requests
Fixed bad ASCIIDOC
Muted test .
[ Zen2 ] Only bootstrap a single node ( )
move to trace level debugging when sigar fails to load
Merge pull request elastic / elasticsearch from rjernst / plugin_default_config
Merge pull request from ywelsch / feature / multi - alias - rest
[ TRANSLOG ] Return null from read ( ) if location generation is outdated
Merge pull request from s1monw / remove_more_deps
execution service : Don ' t try to send empty bulk requests
Fixing CompletionFieldMapper . isReservedChar ( ) to take all relevant chars in account
[ TEST ] Synchronize searcher list in IndexShardTests
When removing an AutoFollower also mark it as removed . ( )
Disable the use of artifactory in CI ( )
Mapping : Default position_offset_gap to 100
Async search : rename REST parameters ( )
Tests : Mute RiverTests . testMultipleRiversStart .
Merge pull request from xuzha / s3 - encryption
[ DOCS ] Add heading and anchor to reindex with ingest pipeline section ( ) ( )
Move CCR REST tests to ccr sub - project ( )
Also short circuit within segment when score mode is avg
Refactor ScriptType to be a top - level class .
Added a FsStats . total section that sums up all stats for the different path . dirs . Added ` disk_io_size ` and ` disk_io_op ` to the toXContent output , summing up ` disk_read_size + disk_write_size ` and ` disk_writes + disk_reads ` respectively .
If no specified index or alias exists and ` ignore_indices ` is set to ` missing ` an index missing error is returned instead of resolving to all open indices ( e . g . when searching ) . This breaks backwards comp . with 0 . 20 . x and before .
Fix discovery - azure - classic plugin packaging test
Test : MinimumMasterNodesTests . testCanNotBringClusterDown didn ' t check for cluster health properly
[ ML ] [ Transforms ] adjusting stats . progress for cont . transforms ( )
Recover small files ( < 1mb ) using a separate threadpool than large files .
remove slf4j license , not used in dist
Adds single node allocation to shrink ( )
add 0 . 90 . 6
remove AwaitsFix from IPHostnameVerification tests now that it is fixed in core
Test : restore logging to prior state
Add Aspire for Elasticsearch to integrations page
IndicesQueryBuilder generates the wrong query name , closes .
fixed checkstyle violation
Merge pull request elastic / elasticsearch from martijnvg / qa / watcher_with_shield
Remove RequestContext from Security ( elastic / elasticsearch )
Check test naming conventions on all modules
Introduce yamlRestCompatTest task and plugin . ( )
Build : Fix Java9 MR build ( )
Does not support " script " in value_clunt aggregation .
Added , that ` body ` is required in " mget " API
Remove readFrom from ingest
[ TEST ] Set proper version on InputStream
remove MapBinder guice binding for processors , use ProcessorsRegistry instead
No more using ' hybrid mmapfs / niofs ' ( )
Build : Add javadocs jars
[ DOCS ] Add ML geographic functions ( elastic / x - pack - elasticsearch )
Call initialising constructor of BaseTasksRequest ( )
ThreadPool : Refactor into several pools , with configurable types per pool , closes .
improve lambda syntax ( allow single expression )
[ HLRC ] Added support for Follow Stats API ( )
Docs tests : cat / health can have max_task_wait_time
Run refresh before executing non realtime get
Remove pluggability of ElectMasterService ( )
Add warning phrase suggester ' s max_errors
Add runtimeJavaDetails property in BuildParams ( )
Internal : fixed wrong version check s / onOrBefore / before in TransportShardSingleOperationAction
Internal : Fix field mappers to always pass through index settings
Merge pull request from ywelsch / fix / doc - tribe
pull out constant for default 5 minute idle setting
Do not set vm . max_map_count when unnecessary ( )
Mute KerberosAuthenticationIT
Merge pull request from MaineC / bug - fix / 15860
Upgrade to 7 . 4 . 0 - snapshot - 1ed95c097b ( )
[ docs ] Update docs for version 2 . 0 . 0 - beta1
SQL : Compact the code in MathOperationTests ( )
master is now 5 . 0 . 0 - alpha5
Merge remote - tracking branch ' origin / master '
Merge pull request from rmuir / nuke_unused_deps
Excpetion = > Exception
Security : remove wrapping in put user response ( )
Allow AD realm to perform ' run - as ' lookups ( elastic / x - pack - elasticsearch )
Merge pull request from rmuir / capturingReferences
[ ML ] Reduce log noise in BaseMlIntegTestCase ( )
ensure no initializing shards during cluster cleanup ( )
Move CharArrays to core lib ( )
Added method that accepts a map as script parameters
Cleanup AbstractSnapshotIntegTest ( )
[ DOCS ] Document get pipeline API as multi - target ( )
Update version skip after backport
[ ML ] Always refresh dest index before starting analytics process ( )
Tests : Add TestZenDiscovery and replace uses of MockZenPing with it ( )
Merge pull request from martijnvg / percolator / filter_by_now
CouchDB River : Deletes not handles correctly
Create utility for custom config setup in packaging tests ( )
SQL : Fix ODBC metadata for DATE & TIME data types ( )
Add reindex progress indicator
Use consistent view of realms for authentication ( )
Fixed NPEs caused by requests without content . ( )
Cut significant_terms to registerAggregation
ldap and ad realms should obey the verification_mode setting ( elastic / elasticsearch )
Use $ { exception_var } for the case of nested generated try / catch
fix repository update with the same settings but different type ( )
Update indexing - speed . asciidoc ( )
Plugins : fix error message while copying plugin config dir
Fix BWC distribution resolution errors ( )
EQL : Use Point In Time inside sequences ( )
Require timeout units when parsing query body
Optimize GCS Repo Uploads ( )
Add roles and cluster privileges for data frame transforms ( )
Added experimental annotation to the update - by - query and reindex docs
Revert " Adds tests for cardinality and filter aggregations ( ) "
[ TEST ] call yaml client close method from test suite ( )
[ ML ] Get Filters should use executeAsyncWithOrigin ( elastic / x - pack - elasticsearch )
createTask can return null
[ DOCS ] Add experimental tag to data frame analytics APIs ( )
Mute testRetentionWhileSnapshotInProgress ( )
[ Test ] Remove ParsedAggregationTests ( )
Search : Add search type ` scan ` allowing to efficiently scan large result set , closes .
[ DOCS ] Adds missing testenv attribute ( )
Add AwaitsFix to KnownActionsTests # testAllTransportHandlersAreKnown
Fix number of nodes in discovery disruption tests
[ ML ] [ Data Frame ] Add support for allow_no_match for endpoints ( )
Add version 7 . 4 . ( )
Harden painless test against " fun " caching ( )
Add read / writeOptionalVLong to StreamInput / Output ( )
Merge pull request from achow / achow / omit - old - current - stats
expose the classic analyzer and the classic tokenizer ( issue )
Don ' t spin up a whole node to unit test some data structures ( )
Java API : XContentFactory creation of XContentBuilder to always be " safe " , closes .
Persistent Tasks : force writeable name of params and status to be the same as their task ( elastic / x - pack - elasticsearch )
Adjust CCR Actions after RequestBuilder is removed
filtered collector should set the next reader on the wrapped collector
Format RemovePluginCommand to 100 - column limit
Improve slow logging in MasterService ( )
Increase timeout in PrimaryFollowerAllocationIT
inner hits : Make sure ` size ` of ` 0 ` works on the ` inner_hits ` level .
[ TEST ] Add skip for 5 . x BWC tests for custom filter in analyze API
Remove processors setting ( )
Docs : Linked to fielddata formats from core types
Simplify handling of git revision ( )
Add admonition to cluster state instability note ( )
[ TEST ] Added cat . allocation and cat . aliases tests
Rename AtomicFieldData to LeafFieldData ( )
Remove hacky node version check in NodeInfo
Also catch NoSuchFileException
cutover more Analyzer . tokenStream to try - with - resources
[ ML ] Rename input_fields to column_names in file structure ( )
Internal : Remove SpawnModules
Adjust BWC version on settings upgrade test ( )
Liberalize StreamOutput # writeStringList ( )
[ DOCS ] Document ` toJSON ` function for role query ( )
Throw MapperParserException if trying to parse value as object .
[ TEST ] Add basic upgrade test script
Change to support name change of email alert action .
Add test for index with two types ( elastic / x - pack - elasticsearch )
Settings : Add secure versions of SSL passphrases ( elastic / x - pack - elasticsearch )
Fixes randomDateTimeZone method ( )
Update keyword . asciidoc
Revert " Muted test . "
1 . 3 . 3 release : add 1 . 3 . 4 version constant
[ Test ] testVerifyApiBlocksDuringPartition - wait for stable cluster after partition
[ Transform ] Replace transforms with transform , index constants ( )
[ DOCS ] Add data stream overview and intro ( )
Revert " Add trace logging when aquiring and releasing operation locks for replication requests "
protected from changing teh number of shards in update settings
convert FilePermissionsTask . groovy to . java ( )
SQL : Move special joda time handling into DocValueExtractor ( elastic / x - pack - elasticsearch )
Fix input snapshotting error when using test clusters cliSetup ( )
move security ingest processors to a sub ingest directory ( )
Docs : Fix encrypt watcher sensitive data documentation ( elastic / x - pack - elasticsearch )
Validate API : support for verbose explanation of succesfully validated queries
Improve error message if setting is not found
Make recovery source partially non - blocking ( )
Remove duplicate operator enums
S3 Repo documentation
Merge pull request from jpountz / upgrade / lucene - 5 . 2 - snapshot - 1674183
[ DOCS ] Relocate discovery module content ( )
Test fixtures improovements ( )
Ignore date ranges containing ' now ' when pre - processing a percolator query ( )
Test : Added more assertions
[ TEST ] mute CoreWithSecurityClientYamlTestSuiteIT
Fix failing tests after merge
Remove deprecated exceptions ( )
Reindex isn ' t a plugin
[ DOCS ] Add collapsible sections to 8 . 0 breaking changes [ Part 2 ] ( )
Throw exception if decay is requested for a field with multiple values
Remove reroute with no reassign
Fix analysis - kuromoji to not use shaded APIs
[ ML ] Do not serialize rules when empty ( elastic / x - pack - elasticsearch )
Add clarifying comment on disrupted in - flight ops
Ignore tabs in sql test keystore files ( elastic / x - pack - elasticsearch )
Fail shard if search execution uncovers corruption
renamed yaml tests
[ ML ] Configure autodetect with scheduled events ( elastic / x - pack - elasticsearch )
Added connection / read timeout support for HTTP
Removed unnecessary get call
[ CLIENT ] Remove unnecessary intermediate interfaces
Add delete autoscaling policy API ( )
Suggester : add suggestRequest to Requests and fix broken javadocs in client
Cleanup HTTP and transport security settings
Refactor script for RC creation
Mute sql . client . VersionTests suite ( )
Compile each Groovy script in its own classloader
Split Tests out of SharedClusterSnapshotRestoreIT ( )
Fixed issue where parent & child queries can fail if a segment doesn ' t have documents with the targeted type or associated parent type
Test : Mute failing SSL test
lucene 4 : field visitors shouldn ' t return fields that were not present in the visited document
Throws IndexNotFoundException in TransportGetAction for unknown System indices ( )
tiny typo fix ( )
Add unittest for GeoPoint seriazliation and corresponding writeGeoPoint method
[ DOCS ] Replaced settings with links ( elastic / x - pack - elasticsearch )
Merge pull request from rothvaw / patch - 1
Explicitly require that derived API keys have no privileges ( )
Do not refresh realm cache unless required ( )
Default ec2 endpoint is ec2 . us - east - 1 . amazonaws . com ( )
[ TEST ] Add automated tests that were missing .
Squash the rest of o . e . rest . action
[ ML ] [ Transform ] unmute bwc tests ( )
Histogram Facet : Add ability to define bounds ( from / to ) to both improve performance and provide additional bound filtering , closes .
[ DOCS ] Adds empty snapshot_id description to revert snapshot API docs ( )
[ DOCS ] Fix analyzer page titles ( )
Merge pull request from jpountz / upgrade / lucene - 5 . 4 - snapshot - 1708254
Add missing word to terms - query . asciidoc ( )
Removing unused AllocationExplanation class ( )
[ DOCS ] Expanding Graph API reference content here & removing it from x - pack - kibana .
ccr : Added maximum translog limit that a single shard changes response can return .
Added support for aliases to create index api
Remove PROTOTYPE from MLT . Item
Permit searches to be concurrent to prewarming ( )
release search context after the caller listener is invoked
[ ML ] Put ML filter API response should contain the filter ( )
Added types exists api
Resolve wording inconsistency
Make the per partition max anomaly scores a result type ( elastic / elasticsearch )
[ TEST ] Make BulkTests # testBulkProcessorFlush more robust
[ DOCS ] Enables testing for monitoring examples ( )
Treat big changes in searchCount as significant and persist the document after such changes ( )
Fix random failures in SearchResponseMerger # testMergeSearchHits ( )
Added a ` scheme ` option to the ` http ` input that supports the values http or https .
Merge pull request from dongjoon - hyun / fix_typos_in_devtools
TESTS : Fix the alert stats test failure and some cleanup and docs .
Make TermVectorService static
fix field names to be disabled again pre 1 . 3
improve numeric stream analysis performance
[ Transform ] remove wrong test ( )
Expose external refreshes through the stats API ( )
Move range field mapper back to core
Fixed serialization error . POST_RECOVERY is now also serialized
Deprecate local param in get_mapping . json ( )
Fix toXContent of GeoShapeQueryBuilder
Abort sorting in case of local agg sort queue overflow ( )
compress metadata stored in gateway
[ TEST ] Fix security test blacklist for new ML test
Enable TRACE for testRecoverBrokenIndexMetadata ( )
Make DocWriteResponse a ToXContentObject
Support requests_per_second = - 1 to mean no throttling in reindex
Always compress based on the settings ( )
[ test ] remove index = from catch : / regex / in date math tests
Adopt to network settings cleanup in elastic / elasticsearch
Use VAULT_TOKEN environment variable if it exists ( )
Remove special case in case no action filters are registered ( )
Fix typo in JavaDoc - - s / note / not
Watcher : Make settings reloadable ( )
Core : Add methods to get locale / timezone in DateFormatter ( )
Ensure output for createClone task
Partition - based include - exclude does not implement equals / hashcode / serialization correctly . ( )
remove hasFetchSubPhaseContext
Internal : an inactive shard is temporarily activated by triggered synced flush
Provide log4j2 logging config for microbenchmarks
Use QUERY_THEN_FETCH ( default ) as search type for data extractor ( elastic / elasticsearch )
Add documentation
Merge remote - tracking branch ' origin / master ' into index - lifecycle
Update to Elasticsearch 0 . 90 . 3 / Lucene 4 . 4 . 0
Muting analytics tests ( )
Do not count shard changes tasks against REST tests ( )
Bump Elasticsearch version to 5 . 0 . 0 - SNAPSHOT
Improve IndexingMemoryController a bit :
Marvel : Add state_uuid to cluster_state documents
Fix parseNamedObject leaving end object token ( )
assert before we set the suppressed
[ DOCS ] Add Data Visualizer to the ML Getting Started tutorial ( elastic / x - pack - elasticsearch )
Remove docs for the removed ` geo_distance_range ` query .
throw a proper exception when no type is found
Merge pull request from elastic / feature / translog_checkpoints
[ DOCS ] Add tutorials section to analysis topic ( )
Change manual stop semantics .
[ TEST ] Fix SearchStatsTests to have all shards allocated
Improves the way the get mapping and get warmer get their data from the master ' s cluster state copy .
REST tests : Bulk API expects an array not a hash
Merge pull request from jpountz / fix / validate_mappings_on_master
Be consistent with the default value of ` acceptable_overhead_ratio ` .
Replace IndexAlreadyExistsException with ResourceAlreadyExistsException ( )
Log Blocked IO Thread State ( )
Replaced hardcoded boolean value with correct parameter
EQL : Sequence / Join parsing and model ( )
fix comment
Change response format of term vector endpoint
Make discovery - azure work again
Split Painless AST into a " user " tree and an " ir " tree ( )
Test : make test less evil to not cause OOME
Remove an unused constant in PutMappingRequest .
Add test for fixed executor rejected count
Remove dead fielddata code .
Adds ignore_unmapped option to nested and P / C queries
Fix put mapping request validators random test
Merge pull request from jbarthelmes / master
Decrease leader and follower check timeout ( )
[ DOCS ] Adds AUC ROC classification metric to the API examples ( )
[ TEST ] fix PhraseInIndex test bug for shard failure
[ tests ] Fix exit code check for systemctl
Upgrade microbenchmarks to JMH 1 . 15
Test : increase security index wait times
Slash miss in indices . put_mapping url
Scripting : split out compile limits and caching ( )
[ DOCS ] Retitle analysis reference pages ( )
Tests : Add rolling upgrade tests for watcher ( )
[ ML ] [ Data Frame ] adjusting change detection workflow ( )
Transpose expected and actual , and remove duplicate info from message . ( )
Merge pull request from nik9000 / build
remove nb structures
Always consume the body in has privileges ( )
[ DOCS ] Shortens ML API intros
[ Docs ] Update ingest - node . asciidoc ( )
Merge pull request from rjernst / integ_test_java_opts
Documentation notes for Range field histograms ( )
Change version in serialization code to 7 . 4 . 0 ( )
Update suggesters . asciidoc
HLRC : Add ML Get Records API ( )
Refactor file script tests to be real unit tests
[ DOCS ] Fixing cross doc link to Stack Overview security topic .
don ' t represent site plugins with ' null ' anymore
[ Rollup ] Remove builders from TermsGroupConfig ( )
Remove deprecated endpoints of hot threads API ( )
Create a thread dump if ES fails to start in packaging tests ( )
Shield Docs : Updated Marvel config and license management for 2 . 0 .
fixes integ test when running as part of the build
Fix message content in users tool ( )
Aliases : Validate alias filter before adding it , closes .
[ DOCS ] Removes redundant Active Directory realm settings ( )
Raise a phase failure if fetch phase gets rejected
add a simple bulk processor to simplify using the bulk API
Internal : fixed typo in NodeService s / disovery / discovery
[ build ] Log more information when es doesn ' t start
[ DOCS ] Add community supported MSI installer to docs
Inline ShardStateAction # innerShardFailed method
Rationalise fetch phase exceptions ( )
Shield Docs : Fixed broken xrefs .
[ cleanup ] renaming classes
[ Doc ] Add a chart about the relative error of the percentiles aggregation .
final
Add support for filtering aliases to MoreLikeThis
Fix signature of createComponents after addition of script service in core
Remove unused ClusterService dependency from SearchPhaseController ( )
Improve connection closing in ` RemoteClusterConnection ` ( )
Merge pull request from colings86 / docs / searchRequestBreakingChanges
[ DOCS ] Remove unneeded comma from CSV processor example ( )
remove the term ' system ' from indicies doc ( )
State default shard limit is not a recommendation ( )
Closes elastic / elasticsearch - Remove marvel_ prefix for all doc types
Round load avg in _cat / nodes .
Simplify handling of fatal network layer errors
Upgrade to mvel 2 . 1 . 1 , closes .
Fixed NumberFormatException in the rest layer by only setting batch_size if it has been specified .
Introduce new put mapping action for dynamic mapping updates . ( )
fix buildscript repos to have jcenter for silly " needed " perforce dep
add type parameters to emptySet ( )
Remove test for closing ingest processors , this is now handled at the
Update to elasticsearch 1 . 3 . 0
Update vector similarity examples to avoid negative scores . ( )
Test : Fix hdfs test fixture setup on windows
Relax regex for getting C + + build so it works for developers with uncommitted changes
Test : fixed test issue
Fix dangling comma in ClusterBlock # toString
[ TEST ] Call get on update job action from DetectionRulesIT
Improved test
Fix north pole overflow error in GeoHashUtils . bbox ( ) ( )
Add destrictive encapsulating all destructive packaging tasks ( )
Temporarily set BWC version to 6 . 0 . 0 for seq . no
Merge pull request from jasontedor / prioritized - runnable - age
Make Recovery API support ` detailed ` params ( )
Unmute rolling upgrade watcher tests and ( )
more accurate chunk size
Range filter gt is broken , closes .
Randomly use Netty 3 plugin in some tests
Upgrade to Lucene 5 . 0 . 0 - snapshot - 1637347
Build : add aggs - matrix - stats to license mapping and ignore sha list ( elastic / x - pack - elasticsearch )
Use RUNTIME_JAVA_HOME for micro benchmarks ( )
Disable BWC tests during backport ( )
Test : Fix repro line for platformTest ( )
[ Docs ] Update multi - search . asciidoc ( )
add getTerms for uid filter
Remove assert statements from field caps documentation . ( )
Fix test jar to contain Mock classes that were moved in 71a3bdb
Always auto - release the flood - stage block ( )
mini layout change to pom . xml
Merge pull request from s1monw / upgrade_on_any_version
Merge pull request from s1monw / add_build_hash_to_headers
Filter on node id in AllocationIdIT ( )
Cli : Simplify test terminals
UpdateByQueryRequest to implement IndicesRequest . Replaceable rather than CompositeIndicesRequest
Remove duplicated javadoc ` fieldType ` param
Watcher : Remove unused test helper class
Fix Path comparisons for Windows tests ( )
Add Kibana upgrade procedure to Upgrade API ( elastic / x - pack - elasticsearch )
Cover missing case in top_metrics test ( )
Fix synchronization in ByteSizeCachingDirectory ( )
[ TEST ] build index up - front to prevent relocations
Bump version to 6 . 0 . 0 - beta1
median absolute deviation agg ( )
Fixed regex in cat . recovery REST tes
Rename transport service method to clarify a user is always used , but
Fix retrieving data stream stats for a DS with multiple backing indices ( )
NETWORKING : MockTransportService Wait for Close ( )
[ Gateway ] set a default of 5m to ` recover_after_time ` when any to the ` expected * Nodes ` is set
Delete by query test on low free disk block ( )
[ TEST ] MLT Rest test needs a mapping since we randomized number of nodes / shards
Update JMH to 1 . 17 . 3
Mute DeprecationHttpIT testDeprecationMessagesCanBeIndexed ( )
update bootstrap startup actions
Test : InternalEngineTests . testSimpleRecover should close snapshot
[ Docs ] Formatting tweaks
monitoring ui : add overall and plugin statuses to ops data
Remove joda time mentions in documentation ( )
test : fix test bug
Queries . calculateMinShouldMatch returns the number of " min should match " clauses that the user wanted
HLRC : Add Get Lifecycle Policy API to HLRC ( )
Unmute PermissionsIT test and enable debug logging for it ( )
Fix jvm - example assertions in packaging tests
Add the ability to define search response listeners in plugins ( )
add more tests and apply feedback from @ mikemccand
Add support for a policy file ( x - pack - extension - security . policy ) in an x - pack extension
Aggregations : The nested aggregator should iterate over the child doc ids in ascending order .
Merge pull request from cbuescher / feature / query - refactoring - spannot
enforce DirectoryReader when globals are loaded
add filter cache tests
Move not a statement responsibility to user tree expression nodes ( )
Fixed bad YAML in msearch test
Simplify ScriptModule and script registration elastic / elasticsearchelastic / elasticsearch ( elastic / elasticsearch )
[ cloud - aws ] Update AWS SDK to 1 . 10 . 12
Address some CCR REST test case flakiness ( )
[ SPEC ] Renamed snapshot / restore apis
Fixes the Version constants for 5 . 2 . 0 , 5 . 2 . 1 , and 5 . 3 . 0 to
Upgrade Maven Surefire Plugin to 2 . 16
Merge pull request from rjernst / pr / mapper - names
release tool : fix ' log exists ' check to not create the log before it checks ; )
Remove all date ' now ' methods from the Painless whitelist to ensure
Advance checkpoints only after persisting ops ( )
[ DOCS ] Adds filter aggregation example link to painless examples ( )
[ DOCS ] Cleans up monitoring configuration steps ( )
Optimize API : Allow to optimize index / indices . Closes .
Exists API can send response body , closes .
[ DOCS ] Document ` index . queries . cache . enabled ` as a static setting ( )
Aggregations : Fix ReverseNestedAggregator to compute the parent document correctly .
[ ML ] Handle simultaneous force delete datafeed and stop datafeed ( elastic / x - pack - elasticsearch )
SQL : Improve error message when unable to translate to ES query DSL ( )
Switch internal representation of order to enum
No need for plenty of RAM for plugin utility ( RAM is often already taken by the elasticsearch service running beside )
Deprecate AbstractHlrc * and AbstractHlrcStreamable * base test classes ( )
Add auto create action ( )
Merge pull request from nik9000 / xlint_serial
Aggregations : Fixes Filter and FiltersAggregation to work with empty query
Better test reuse of existing mappings for dynamic mappings .
fix the type of the mapping
Fix buggy test
Error on bad shape relations in runtime fields ( )
fix wrong closing of possible null mappings in geo point mapping
Reformatting source to fit 100 character line length restriction
Fix unconverted ML Action Ctors to use Writeable ( )
[ TEST ] Use search api to retrieve data counts instead of via get job stats api .
[ BUILD ] Allow smoke - testing other artifact but staging artifacts
Remove plugin isolation feature for a future version
Use the elasticsearch . bat file in zip distribution ( )
testRejectingJoinWithIncompatibleVersion ( WithUnrecoveredState ) should use immediate priorities
Add field_stats to kibana user role in shield
Document simple_query_string negation with default_operator of OR
SQL : Upgrade to Joda 2 . 9 . 9 ( sync with master )
fixed the watcher info API
Disable BWC for script cache settings change ( )
add completed count to thread pools
add lang - python module
Fix typo in comment in o / e / c / c / Cache . java
SQL : Fix KnownActionsTests after introduction of the _sql / translate endpoint
SQL : Support larger strings in binary protocol ( elastic / x - pack - elasticsearch )
Fix defaults in GeoShapeFieldMapper output ( )
[ TEST ] Reduce the amount of docs being indexed .
completion_fields and field_data fields not support in node stats yet .
Add the fields option to the search API docs . ( )
Adjust BWC version for discovery node roles
Uses TransportMasterNodeAction to update shard snapshot status ( )
Add Debug / Trace logging to token service ( )
Add logstash system index APIs ( )
test
Remove random when using HLRC sync and async calls ( )
[ Discovery ] lightweight minimum master node recovery
Fix search_as_you_type ' s sub - fields to pick their names from the full path of the root field ( )
better containing of hadoop for actual blobstore operations
Build : Convert rolling upgrade test to create tasks per bwc version ( elastic / x - pack - elasticsearch )
[ DOCS ] Added X - Pack bootstrap checks ( elastic / x - pack - elasticsearch )
Merge pull request from nik9000 / fix_azure_vagrant
Add more def tests including checking bytecode for specific
Don ’ t ack if unable to remove failing replica ( )
Internal : Remove interfaces for auth services
Add Tests for findSamlRealm ( )
Tribe : Index level blocks , index conflict settings
Merge pull request from elastic / markw / typo - parentchild - releasenotes - 14508
Disable rhel8 in packaging tests ( )
Made sure that named filters and queries defined in a wrapped query and filter are not lost .
add search hit phase , which operates on a hit , and have all relevant hit phases implement and use it
Do not cache script queries . ( )
Mute test CompletionSuggestionTests . testToReduce
test : removed the ' plugin . types ' usages from the source code
[ DOCS ] Deterministic scripted queries are cached ( )
Threadpool Info : Allow to serialize negative thread pool sizes
use read / writeOptionalWriteable
[ DOCS ] Update get data stream API
docs : removed obsolete information , percolator queries are not longer loaded into jvm heap memory .
[ DOCS ] Remove references to changelog and to highlights
[ DOCS ] Add links to ` flattened ` datatype ( )
Display list of all available site plugins on / _plugins / end point fix
[ DOCS ] Fixed cross doc links .
Do not preallocate bytes for channel buffer ( )
Revert " [ DOCS ] Re - enabled graph API example testing . "
Use m_m_nodes from Zen1 master for Zen2 bootstrap ( )
Disallow merging existing mapping field definitions in templates ( )
add TODO optimizations comments
Add tests for highlighting boost query .
[ ML ] [ Data Frame ] adds index validations to _start data frame transform ( )
Analysis : Expose light and minimal language token filters , closes .
Re - enable doc testing for Pipeline Aggregations ( )
SLM PUT : add precision on date math support in indices parameter ( )
Adjust reload keystore test to pass in FIPS ( )
Revert " Refactor AllocatedPersistentTask # init ( ) , move rollup logic out of ctor ( ) "
Pass tests . jvms system property to test tasks for maxParallelForks ( )
Made for / each break tests more robust in Painless .
remove unnecessary unboxing , we do that as impl detail in parseTika
Use custom index metadata for ILM state ( )
Remove some line length supressions ( )
[ ML ] Adjust memory limit test to account for greater accuracy ( elastic / x - pack - elasticsearch )
Failure to reduce geo distance sorting ( class case exception ) , closes .
Create temporary directory if needed in CCR test
Example of how to set slow logs dynamically per - index ( )
Add abstract TestScript to the test framework jar
more work on maven
Use Elasticsearch IOUtils ( elastic / x - pack - elasticsearch )
Painless : Fix regex lexer and error messages ( )
Fix logic in dockerComposeSupported ( )
allow to perform operations on a recovering shard using versioning to do conflict detection
[ maven ] rename maven names / ids for distribution modules
Update health . asciidoc
Fix typos in comments / strings of ` test ` module .
Disable logger usage checks
Update link to mapper - murmur3 plugin in card docs ( )
lucene 4 : fix TTL
Fix ML upgrade tests
Add ` prefer_local ` flag to analyze and percolate request , closes .
Remove MathUtils .
Plugins : Add plugin extension capabilities ( )
Reindex : do not log when can ' t clear old scroll ( )
Only re - parse operation if a mapping update was needed
Remove support for ES_INCLUDE
[ DOCS ] Fix Gsub processor snippet ( )
auto_expand_replicas causing very large amount of cluster state changes when a node joins or leaves the cluster
Adding ML HLRC wrapper and put_job API call ( )
[ DOCS ] Add transforms to build . gradle
Replace deprecated API usage in Netty4HttpChannel
[ DOCS ] Remove Wikipedia link from ` SUM_OF_SQUARES ` SQL function docs ( )
Add more reporting to asserting index searcher
Removed flaky test . Looks like randomisation makes these assertions unreliable .
force check after X bytes indexed
Handle empty input in AddStringKeyStoreCommand ( )
add logging when trying to obtain node lock
Upgrade to Lucene 3 . 5 , closes .
Ensure domain_name setting for AD realm is present ( )
Painless : Move and Rename Several Methods in the lookup package ( )
Remove Support for VERSION_CHECKPOINTS Translogs ( )
Ensure no ongoing peer recovery in translog yaml test ( )
Testing : Change qa / vagrant artifactId
Tests : Fix randomizedtest fail
Don ' t update nodes list when stepping down as master ( )
[ DOCS ] . Security index is never auto created ( )
[ DOCS ] Moves analysis resources to PUT DFA API docs ( )
Suggest : Move name of suggestion to SuggestBuilder
Add more tests for Azure Repository client selection
Date Histogram Facet : Add ` quarter ` as an interval , closes .
[ ML ] [ Data Frames ] unify validation exceptions between PUT / _preview ( )
Core : refactor upgrade API to use transport and write minimum compatible version that the index was upgraded to
[ test ] mute IndexShardTests . testDocStats
make test assertions relative in IsolatedPluginTests
Add log message for auto - follower timeout
Update max_concurrent_shard_request parameter docs ( )
Version option should display if snapshot
add anotehr test
Closing a ReleasableBytesStreamOutput closes the underlying BigArray ( )
Redocument the ` index . merge . scheduler . max_thread_count ` setting
[ ML ] Put back check for inconsistent job id
Revert " Fix elasticsearch - cli dependency "
Make synced flush attemp async to run it easily from a TransportAction
Fix typos in query dsl docs .
Expose Lucenes Ukrainian analyzer
refactor script execution handling to improve performance , also introduce specific search script executor
Skip alias name validation rest test on 5 . 0 cluster
Build : Cleanup test classes in buildSrc to be less groovyish
Build : Give better output for java version introspection ( )
test : unmuted slow watches test .
Increase timeout in ILM doc test slightly ( )
TransportVerifyShardBeforeCloseAction should force a flush ( )
[ ML ] Reject put datafeed for job that is marked as deleted ( elastic / x - pack - elasticsearch )
Use allocation ids to prevent repeated recovery of failed shards
Disable BWC tests for backport of ( )
Only enable soft - deletes in 6 . 5 or later
Descriptions for scripting enabled ( )
Mute failing REST tests
[ DOCS ] Add Java to list of HTTP client libraries for basic authentication ( )
Add missing license file to ThreadPoolTypeSettingsValidatorTests . java
Merge pull request from xuzha / xu - retry - s3snapshot
[ TEST ] Provide overloaded indexRandom to opt out of dummy documents
lazily allocate the LZF buffer if its only needed
license extractor
[ ENGINE ] Close Translog if initial commit fails
automatically clean indices that no longer exists in the metadata , this can happen if we delete a closed index , or when a node joins a cluster , that no longer has those indices
Remove JDK 9 flag on CI builds
[ DOCS ] Document ` nodes ` cluster stats ( )
speed up slow test
Removed pipeline tests with a simpler tests
Adjust skip version after backport ( )
SCRIPTING : Move Aggregation Scripts to their own context ( )
Mark x - pack plugin as having native controller
xpack plugin can be isolated now that we have xpack extension support
Implement aggregations on aggregate metric fields ( )
[ TRANSLOG ] Fold Translog . View into it ' s only implementation
Streamline support for get / set / remove of metadata fields and ingest metadata fields
Merge pull request from nik9000 / no_windows_deb_tests
[ Transform ] implement throttling in indexer ( )
mute NamedDateTimeProcessorTests
Parent / Child Support , closes .
[ Benchmark ] Fix TermsAggregationSearchBenchmark : The ` ordinals ` execution mode doesn ' t exist anymore .
Throw exception on duplicate mappings metadata fields when merging templates ( )
default socket reuse address value should be non null
settings : Make ` action . auto_create_index ` setting a dynamic cluster setting .
[ SPEC ] Added the routing and timeout params to bulk
Parse script on storage instead of on retrieval
[ refactor ] add Environment in BootstrapContext ( )
Check bucket metric ages point to a multi bucket agg ( )
doc
replace custom functional interface with CheckedFunction in percolate module
Move alias metadata from settings to a separate data structure
[ DOCS ] Remove outdated file scripts refererence ( )
fix compile error after https : / / github . com / elastic / elasticsearch / pull / 21032
[ TEST ] In PercolatorTests # testPercolateSorting_unsupportedField we need to be sure that the . percolator type exists
rename some histogram collectors
Core : In the bitset cache only eagerly load bitsets for parent nested object fields .
Test : write heap dump per default
Clean up BytesReference ( )
Internal : move to lucene BoostQuery
Fixes broken links .
Remove assemble from build task when assemble removed
Merge pull request from rjernst / less_forbidden
added put , get and delete pipeline APIs .
Watcher : migrate PagerDuty v1 events API to v2 API ( )
time value to support parsing days as well
Remember the index each result came from ( elastic / elasticsearch )
Add missing await into an interruptable block
[ ML ] Make datafeeds run - as the user who created / updated them ( elastic / x - pack - elasticsearch )
Segments API : Support merge id on segments ( groups segments being merged )
remove . idea
streams to handle BytesReference as a generic type as well
[ DOCS ] Use a better name for fields in examples to avoid ambiguity
adapt changes from upstream
Docs : Added RethinkDB River to list of community plugins
Correctly determine defaults of settings which depend on other settings ( )
Merge pull request from cbuescher / feature / query - refactoring - fquery
SQL : Do not resolve self - referencing aliases ( )
Do not allow registering basic licenses ( elastic / x - pack - elasticsearch )
Tests : make randomRepoPath work with bwc tests
small refactor
Check dot - index rules after template application ( )
always debug the user dn role mapping
Add support for timestamp field
Add a scripted similarity . ( )
Only log vagrant output on failure ( )
Fix reindex under the transport client
[ DOCS ] Note that DS timestamp field mapping changes require reindex ( )
CORS : Support regular expressions for origin to match against
ensure that modificatons to the enum order trigger test failures since we rely on the ordinal
Enable adaptive replica selection by default ( )
Fix broken NaN check in MovingFunctions # stdDev ( ) ( )
Fix querying a data stream name in _index field . ( )
Merge pull request from jimferenczi / disable_stored_fields
[ DOCS ] Reformat condition token filter ( )
Create standard gc and memory_pool names for Jvm stats
Remove ` node . mode ` and ` node . local ` settings ( )
Java api : add missing boost support to GeoShapeQueryBuilder
[ TEST ] mute IndicesStoreIntegrationTests . indexCleanup
Fix issue with finishing handshake in ssl driver ( )
[ ML ] Also revert quantiles and model_size_stats ( elastic / x - pack - elasticsearch )
Add get / exists method to RestHighLevelClient ( )
[ Test ] Add test for custom requests in High Level Rest Client ( )
Merge pull request from jimferenczi / did_you_mean_plugin
added breaking docs
Updated actions list for 2 . 0 . Closes elastic / elasticsearch .
update to master
Docs : Adding community - contributed Scout plugins
Speed up date_histogram by precomputing ranges ( )
Make tmp file handling for jjs scripts better
Update Settings : Allow to dynamically change refresh_interval and merge policy settings , closes .
Index folder names now use a UUID ( not the index UUID but one specific
[ TEST ] return empty array if AbstractQueryTestCase # currentTypes is null
[ CORE ] Renamed internal action to be consistent
Packaging : Install all plugins during bats tests
change logging level of testTimeoutSendExceptionWithDelayedResponse to TRACE
prepare release elasticsearch - analysis - icu - 2 . 0 . 0 . RC1
Use the special java . ext . dirs path instead of 4 tomcat heuristics
Watcher ' s ExecutionService is not thread - safe . ( elastic / x - pack - elasticsearch )
Merge pull request from uschindler / painless_arrayindextype
Merge pull request from clintongormley / version_bump_master
Highlighting : Using " plain " ( non term vector ) highlighting with custom score within a filtered query fails to highlight , closes .
Add remote cluster client ( )
Follow up for elastic / elasticsearchelastic / elasticsearch ( elastic / elasticsearch )
Fix ids query builder test to use the same logic for matchnodocs as the
Fill LocalCheckpointTracker with Lucene commit ( )
Merge pull request from nik9000 / unflake_test
Use TestEnvironment factory method to create test Environment objects ( elastic / x - pack - elasticsearch )
Don ' t build packages on non Linux ( )
better failure message when failing to get a node lock
Add ` http ` and ` transport ` simplified host settings , closes .
Merge pull request from rjernst / fixtures
[ BUILD ] fix snapshot URL
Emphasize that filesystem - level backups don ' t work ( )
Geo : refactor geo mapper and query builder ( )
Mute EmailSslTests test case in fips ( )
Update TLS configuration in Docker docs ( )
Expose multi - valued dates to scripts and document painless ' s date functions ( )
TESTS : Fix Concurent Remote Connection Updates ( )
[ ml ] Don ’ t open failed jobs after restart ( elastic / x - pack - elasticsearch )
Catch exception in callback of exception
Merge pull request from colings86 / deprecate / indicesQuery
Expose CCR to the transport client ( )
Fix quoted _exists_ query ( )
[ DOCS ] Added docs for term vectors
Remove timeout mechanism from ShardStateAction
[ SPECS ] Updated percolator specs .
Remove PrefixAnalyzer , because it is no longer used .
Remove - Xlint : - serial , the crazy exceptions are now gone
Docs : Fixed missing comma in multi match query example
[ ML ] Prevent unnecessary job updates . ( elastic / x - pack - elasticsearch )
make host state immutable
Enforce username validation on PUT security / user ( elastic / elasticsearch )
Test : add awaits fix for SessionFactoryLoadBalancingTests
Do not swallow exceptions thrown while parsing settings
[ Watcher ] Removed not needed usages of ` com . google . common . * ` classes .
Remove AbstractComponent from AbstractLifecycleComponent ( )
Respect ML upgrade mode in TrainedModelStatsService ( )
[ API ] renaming search - template to search_template .
Add response params to REST params did you mean
better caching of missing filter / query
Simplify CacheKey used for scripts
Account for rpm behavior difference on directories
[ Engine ] Verify checksums on merge
Previously added logging in TransportRefreshAction caused NPE if search shards had errors ( or was not yet started / assigned )
fix wrong refactoring . . .
Merge pull request from jpountz / fix / 112_snapshot
Simplify TransportMultiSearchActionTests ( )
[ ML ] [ TEST ] Fix bucket count assertion in ModelPlotsIT ( )
Make suggester implementation pluggable
Snapshot / Restore : restore of indices that are only partially available in the cluster
Merge AwsS3Service and InternalAwsS3Service in a S3Service class ( )
Alerting : Add parsing and indices .
Change missing authn message to not mention tokens ( )
Replace CustomMustacheFactory constant with same constant from Script ( CONTENT_TYPE_OPTION )
improve gateway initial recovery time by creating the indices , and only after all have been created , do the rerouting
awaitFix testCorruptTranslogTruncationOfReplica
Merge pull request elastic / elasticsearch from elastic / modularize_netty
Merge pull request from jdconrad / ref
Update to Lucene 4 . 10 . 0
Resync fails to notify on unavaiable exceptions ( )
Docs : In the code there is the ` d ` option .
Make specifying java home mandatory for testclusters ( )
Prevent random build failures
Merge pull request elastic / elasticsearch from jaymode / request_context
Decouple ClusterStateTaskListener & ClusterApplier ( )
Add version and create_time to data frame analytics config ( )
Reverting 127 . 0 . 0 . 1 to localhost
Update the version compatibility for zero_terms_query in match_phrase .
Indices resolution : special treatment for IndicesAliasesRequest
Clarify refresh parameter in the ` _bulk ` API
Remove anonymous PublicationContext implementation ( )
[ DOCS ] Added JavaDocs for ClusterAdminClient , IndicesAdminClient and Warmer API
introduce 7 . 2 . 2 as a version ( )
[ DOCS ] Delayed data annotations ( )
Log warning when unlicensed realms are skipped ( )
[ DOCS ] Add FAQ for ML rolling upgrade issue ( elastic / x - pack - elasticsearch )
Shade external dependencies in the rest client jar
Fix build errors ( )
Makes the same_shard host dyanamically updatable ( )
nextSetBit should check if the underlaying array contains the current word ( )
[ DOCS ] Reformat refresh API docs ( )
also test terms facet with execution hint of map
Added bwc indices for 2 . 0 . 0 - rc1
Remove redundant configuration in BuildPlugin
Require that relocation source is marked as relocating before starting recovery to relocation target
Version missing a space , closes .
Create snapshot role ( )
[ ci ] Fix bug with - Dtests . jvm . argline
Prevent licenses to be upgraded to production unless TLS is configured ( elastic / x - pack - elasticsearch )
don ' t call clusterService again to get filtering aliases
Add support for alpha versions
some misspelled words in code ( )
remove pom . xml from ingest plugin
Mute data stream setup in remote cluster REST tests ( )
rename more before ( ) methods that now conflict with ESTestCase
Remove RestController from tests where it ' s not needed ( )
Fix scroll query with a sort that is a prefix of the index sort ( )
Made AggregatorFactory fields final and removed AggregationContext from createInternal ( ) parameters
[ DOCS ] Remove fixed file from build . gradle
Merge pull request from bharvidixit / master
Improved transport sniffing docs
MockScriptEngine and associated plugin has changed in ES core
Tests : Use script . max_compilations_rate for BWC tests
lucene4 : make SimpleIdCache more resilient to missing fields
Update FilterBuilder and QueryBuilder interfaces
Mute testHistoryIsWrittenWithDeletion
also support _percolate for bulk item requests
Await all pending activity in testConnectAndDisconnect ( )
Remove get index templates deprecated methods
[ TEST ] Temporarily don ' t corrupt . del files since they are generational and we might pick the wrong one
Stop Copying Every Http Request in Message Handler ( )
[ TEST ] NettyTransportMultiPortTests now checks selected random ports are free before using them
[ GEO ] OGC compliant polygons fail with ambiguity
capture current state of peer level recovery
Defer mapper - attachments for now
Adds unit tests for and removes prior integration tests . Throws exception on redefining built - in similarities only for indices created on or after v3 .
Doc : Clarify that index . routing . allocation . total_shards_per_node means both primary and replica shards
[ DOCS ] Better document the limitation of nested objects .
fix logging
don ' t clear on both obtain and release for the search cache
Add append processor
test : do not schedule when test has stopped
Build : Add pom generation to assemble task
[ TEST ] Redirect links to new locations ( )
[ DOCS ] Various spelling corrections ( )
[ Bulk ] Release semaphore again after close
Clarifies the documentation for the ` http . cors . enabled ` setting ( )
[ ML ] fixes minor snapshot upgrader task failure bug and rest client test failure ( )
[ DOCS ] Fixes broken links ( )
Fix use of apostrophe
HLRC : ML Add preview datafeed api ( )
improve custom name resolver logic to allow more options for control
cleanup formatting in significant aggs package
Plugin cli : Improve maven coordinates detection
Upgrade to Lucene - 7 . 4 . 0 - snapshot - 0a7c3f462f ( )
Removes the old handler for shard snapshot status
Mappings : Remove the ` ignore_conflicts ` option .
Use MapperTestCase for cartesian mappers tests ( )
Reindex : Fix error in delete - by - query rest spec ( )
Fix copy - pasted javadoc error
Fail if ex null
remove dead code
Documented privileges required to use cross cluster search with Security
EQL : Improve retrieval of results ( )
[ DOCS ] Align with ILM API docs ( )
Merge pull request from s1monw / convert_client_settings
GeometryTestUtils should always generate valid polygons ( )
Optimise rejection of out - of - range ` long ` values ( )
[ TEST ] Add test only API that allows to pass the inactive time in NS directly
Scripting : Add StatefulFactoryType as optional intermediate factory in script contexts ( )
Fixing issues after master rebase
lucene 4 : Field name no longed interned when loading field data cache and return empty field data cache for fields that don ' t exist .
Tests : Fixing more tests in Watcher after query refactoring
Fix handling of null values in geo_point ( )
release tool : Improve output for open github issues
Disable BWC tests for backport of
[ CCR ] Make leader cluster a required argument . ( )
Tests : Allow unmapped fields in SimpleQueryStringBuilderTest # testDefaultOperatorHandling
Fix NPE in token_count datatype with null value ( )
Merge pull request from brwe / vagrant - install - groovy
Require / bin / bash in packaging
[ DOCS ] Fix link ( )
Security on Basic License
inner hits : Fix bug that resolves parent docs properly as inner hit when inner hit is defined on has_parent query .
Reduce max javac memory to 512mb ( )
Improve error message for absence of indices ( )
more options on stress rolling restart test , variable number of fields , variable length of text
[ DOCS ] Fix typo ( )
Silence lint warnings in server project - part 2 ( )
[ release script ] Check for / / NORELEASE in code before release
Add support for ` teardown ` section in REST tests
[ TEST ] Wait for shards to allocate before cleaning caches
[ TEST ] Wait for replicas before stopping nodes in ML distributed test ( )
Merge pull request elastic / elasticsearch from jasontedor / hash - be - gone
Deprecate tribe service
[ TEST ] also close wrapped readers
Mute DynamicMappingIT testMappingsPropagatedToMasterNodeImmediately
Adds REST client support for starting and stopping ILM ( )
Mute HttpReadWriteHandlerTests . testReadTimeout
Computational geometry logic changes to support OGC standards
Enable decompression of response within LowLevelRestClient ( )
Exception : TokenStream implementation classes or at least their incrementToken ( ) implementation must be final , closes .
[ UPDATE ] Handle document missing exception on retry correctly .
Merge pull request elastic / elasticsearch from simianhacker / 2 . 1
Use regular Map for attributes in DiscoveryNode , get rid of attributes getter in favour of regular getter
remove v6 . 5 . x and v6 . 6 . x version constants ( )
fixed compile error due to a recent upstream change in core
[ DOCS ] Prepare XPack APIs for Elasticsearch Ref ( elastic / x - pack - elasticsearch )
Fix roundUp parsing with composite patterns ( )
update to 0 . 15 . 0 SNAPSHOT
Replace fixtures with docker - compose ( )
[ DOCS ] Correct required file ext for user agent ingest processor ( )
Mappings : Allow _version to use ` disk ` as a doc values format .
Update completion - suggest . asciidoc
[ CORE ] Fold engine into IndexShard
_count hangs , closes
Suppress LineLength in more files
Update to Lucene 4 . 10 . 1
[ DOCS ] Adds missing icons to Graph HLRC APIs ( )
Merge pull request from mfussenegger / migrate - 2_0_settings
[ DOCS ] Adds cluster setting to enable / disable config migration ( )
[ Rollup ] Metric config parser must use builder so validation runs ( )
Add integration test for Delete - By - Query and Security
Build : Convert xplugins to use new extra projects setup ( elastic / elasticsearch )
[ ML ] [ Data Frame ] Fixing default delay set in timesync ( )
better merge mapping logic , better failure reporting
Documented how to define custom mappings for all indexes and all types
Sort plugins in list x - pack extensions command
Begin migrating SQL ' s next page ( elastic / x - pack - elasticsearch )
SQL : Remove the deprecated ` AttributeMap ( Map ) ` calls ( )
Introduce StepListener ( )
[ DOCS ] Fix search request body links ( )
prepare for next development iteration
TEST : add log testDoNotRenewSyncedFlushWhenAllSealed
Fix Snapshot Finalization not Waiting for Index Metadata ( )
Absorb ImmutableSettings into Settings
Fix line longer than 140 characters in Murmur3FieldMapper
Merge pull request from rmuir / buglevel
Relax a bit the accuracy tests on SloppyMaths .
Node Stats : Certain indices level stats to retain stats even when shard relocates , closes .
Rename CONF_DIR to ES_PATH_CONF
Modify unsigned_long tests for broader ranges ( )
[ DOCS ] Clarify results_index_name description ( )
[ DOCS ] Test framework documentation
Mute AsyncSearchActionIT test ( )
Mute SearchWithRandomExceptionsIT ( )
Remove the test which is testing java and joda api ( )
add nocommits for Booleans parsing leniency
Randomize AllocationDecider order in tests
[ DOCS ] Document ` breakers ` , ` script ` , and ` discovery ` node stats ( )
remove PROTOTYPE from DiscoveryNode
Fix index selection for snapshot restore when non - existent indices are specified ( )
Only refresh id_cache if the parent field is active .
Remove some ceremony in agg parsing ( )
Ensure we protect Collections obtained from scripts from self - referencing ( )
[ test ] Bundle TestShardRouting in test jar
Added version constants for 1 . 1 . 2 and 1 . 0 . 4
Relax history check in ShardFollowTaskReplicationTests ( )
HLRC : Add ML Get Buckets API ( )
move to 0 . 17 . 0 - SNAPSHOT
Share common readFrom / writeTo code in AcknowledgeResponse ( )
Missing import
Fix test base class that disables ML autodetect
Adressing comments
Fix more download URLs
Remove DEBUG - level default logging from actions ( )
Separate acquiring safe commit and last commit ( )
Adapt to getProperty move out of MultiBucketsAggregation . Bucket ( elastic / x - pack - elasticsearch )
ILM : migrate action configures the _tier_preference setting ( )
Enable debug log in testRetentionLeasesSyncOnRecovery
[ TEST ] Move rebalance setting to the index level / pass it on index creation
Fix Active Directory tests ( )
[ DOCS ] Fuzzy wildcard not supported in ` query_string ` ( )
Refactor delete interim results ( elastic / elasticsearch )
Core : Make sure deletes are unscheduled and no npe is returned when a alert is deleted twice
Muted on master
Update api - conventions . asciidoc ( )
Don ' t fail in ` afterExecute ` if context is already closed ( )
fix plugin . bat classpath , closes .
Fix BWC issues for x_pack / usage ( )
Testclusters : improove timeout handling ( )
Exempt LocalDate from bridge method checks on java 9
Add support for distance queries on geo_shape queries ( )
S3 Repository : Eagerly load static settings ( )
[ Test ] Speeup FieldData tests and remove unnecessary @ Repeat annotations
Merge pull request from wikka / sa - east - 1
Docs : Fix : issue : and : pull : asciidoc replacements
Remove trailing comma from nodes lists ( )
[ ML ] renames * / inference * apis to * / trained_models * ( )
[ ML ] Report index unavailable instead of waiting for lazy node ( )
SQL : Fix simplification of boolean expressions . ( elastic / x - pack - elasticsearch )
fix ignore
log this
[ DOCS ] Deprecated the path setting in favour of copy_to
Remove extraneous quotes from Windows env script
[ DOCS ] Add property table for ML Update Jobs API ( elastic / x - pack - elasticsearch )
match processor should handler values other than string properly ( )
security : resolve date match expressions for authorization
Painless : Add PainlessClassBuilder ( )
Merge pull request elastic / elasticsearch from rjernst / api_jar
Remove redundant imports
[ DOCS ] Fix typos in build . gradle comments ( )
[ Zen2 ] Storage layer WriteStateException propagation ( )
[ ML ] Increase timeout for RegressionIT . testAliasFields job to run ( )
Add CumulativeCard pipeline agg to pipeline index ( )
Add Support in geo_match enrichment policy for any type of geometry ( )
Improving and cleaning up tests
use a released version of attached - artifact - enforcer instead of snapshot
[ ML ] Add validation that rejects duplicate detectors in PutJobAction ( )
[ DOCS ] Reformat match query ( )
Remove internal channel tracking in transports ( )
mute test on windows master
Update heap setting documentation in light of machine dependent heap ( )
[ DOCS ] Rewrite aggs overview ( )
Add support for flat_settings flag to all REST APIs that output settings
Fix path for tests
Make sure filter is correctly parsed for multi - term vectors
Fix CoordinatorTests . testUnresponsiveLeaderDetectedEventually ( )
Better test diag output on OOM ( )
Move to standard ES v0 . 90 . 6
Use explicit test classes output dir for configuring test task
[ TEST ] Increase timeout for ML internal cluster cleanup ( )
Pull snapshots prior to 6 . 3 from elastic maven ( elastic / x - pack - elasticsearch )
Pass DiscoveryNode to initiateChannel ( )
Deprecate cert gen without a CA and add self - signed option ( )
Extract proxy connection logic to specialized class ( )
Merge pull request from jpountz / enhancement / per_segment_apis
Use Analyzer . tokenStream ( fieldName , text ) instead of Analyzer . tokenStream ( fieldName , new StringReader ( text ) ) .
Optimize multiple cluster state processing on receiving nodes
allow for concurrent flushing while indexing
Make xpack . monitoring . enabled setting a no - op ( )
Remove Unused Single Delete in BlobStoreRepository ( )
Respect Master Node Timeout for Delete Snapshot Requests ( )
Merge pull request from jasontedor / test - cluster - service - no - timeout
[ Docs ] Correct README example snippet ( )
Upgrade from JMH 1 . 12 to JMH 1 . 14
Do not lose CacheTest failure stack traces
Improve performance when searching across multiple types , closes .
cluster formation DSL - Gradle integration - part 2 ( )
Re - enable BWC tests ( )
Update repositories . asciidoc
Mark filtered query example as not to be used ( )
[ TESTS ] Fix test to use index - level doc IDs instead of segment - level doc IDs .
Fix : Use context instead of headers for storing the auth token
Add histogram field type support to boxplot aggs ( )
Release scripts : Split prepare_release into two scripts
Add 5 . 2 . 0 to static versions used by vagrant
[ DOCS ] Added transport client info for X - Pack ( elastic / x - pack - elasticsearch )
Mute IndexFollowingIT . testDeleteLeaderIndex ( )
Un - deprecate NamedXContentRegistry . Entry constructor that takes a context ( )
Replace magic number in test with random number .
Merge pull request from jasontedor / fix / 12382
[ build ] Fix deb and rpm tests
Update getting - started . asciidoc ( )
Added quotes to allow spaces in installation path
[ TESTS ] Mute SmokeTestWatcherWithSecurityIT tests
Create es - 1 . 2 branch
clean up the java api a bit
Revert " it ' s a ` noop ` operation , not a ` none ` operation . ( ) "
Make merging mappings smarter , closes .
Add usage stats for frozen indices ( )
Tests : Increase loglevel , deactivate ML / monitoring to unclutter logs
Remove more hidden file leniency from plugins
Add documentation for Java API update / get settings API
Consolidate the last easy parser construction ( )
allow index and type to be specified as arrays in MultiSearchRequest
check custom meta data to avoid NPE ( )
Deprecate rivers
[ DOCS ] Changes scripted metric to filter aggs in transforms example ( )
[ DOCS ] Add xref to agg metric double field type ( )
Transport client : Don ' t validate node in handshake ( )
supported field type tests for max agg ( )
docs : update example for finding percolator where query terms couldn ' t be extracted successfully
can now reference single value metrics directly instead of having to add ' . value ' to the path
Give significance lookups their own home ( )
Fix scaling thread pool test bug
Upgrade spotless dependency ( )
[ TEST ] Sort by the _id field instead of _uid field and also assert the sort value .
[ ML ] Allowing _close to accept body payloads for options ( ) ( )
don ' t assert exact expected lengths for documents .
Get lang - javascript , lang - python , securemock ready for script refactoring .
Build : Remove buildSrc as a project in IDEs
Fix NPE / AIOOBE when building a bucket which has not been collected .
Merge pull request from javanna / enhancement / rename_search_service_transport_action
Reference docs fixes
Mute MlDistributedFailureIT # testFullClusterRestart ( )
Mark Log4j API dependency as non - optional
Docs : Mark SQL Geo functionality as beta ( )
Properly fake corrupted translog ( )
Secure password for monitoring HTTP exporter ( )
CONSOLE - ify min and max aggregation docs
Upgrade lucene to lucene - 8 . 7 . 0 - snapshot - 3b59906 ( )
[ DOCS ] More assertive statement about requiring ` query ` in _count , etc
[ doc ] Move mapper attachment plugin to mapper page
Packaging : Ensure upgrade_is_oss flag file is always deleted ( )
Handle 5 . 6 . 6 and 6 . 1 . 2 release
[ DOCS ] Remove unneeded word in EQL docs
Moved json field variable from AlertActionEntry to AlertActionManager and made AlertActionEntry not implement ToXContent .
add snapshot status ( either current one or the latest one executed ) to indices status api
Changes after review .
Core : Removed unused methods .
Some Cleanup in o . e . gateway Package ( )
Mute testFollowIndexWithConcurrentMappingChanges ( )
[ DOCS ] add docs for REST high level client index method ( )
Enhance parsing of StatusCode in SAML Responses ( )
Merge pull request from rmuir / network_config_improvements
Merge pull request from javanna / test / random_search_ext
Rename Task Persistence into Storing Task Results
SQL : Better handling of number parsing exceptions ( )
add extended validation information
[ Docs ] Update daterange - aggregation . asciidoc ( )
[ TEST ] Fix typo in MovAvgIT test
[ Transform ] fix bug in supporting boolean values in pivot ( )
Adjust Lucene versions for 7 . 9 . 1 and 7 . 10 . 0
We should cope with the situation when a watch is going to be executed whilst shutting down .
use Lucene ' s 4 . 10 . 3 version constant
Revert cfd49f8 : type is required in " indices . put_mapping "
[ DOCS ] Fix hyphenation for " time series " ( )
Fix Rest Tests Failing to Cleanup Rollup Jobs ( )
Improve stability of RobinEngineIntegrationTest by reducing direct memory usage of the test env
[ CCR ] Added auto_follow_exception . timestamp field to auto follow stats ( )
Shield : Fix compilation error in IndexAuditTrailTests
Tests : Fix more bad assumptions about routing in TransportTwoNodesSearchTests .
Adding new screenshots and updating copy . ( )
[ Docs ] Fix single page : docs : check invocation ( )
Merge pull request from brwe / exception - on - force - merge
Filtered out non data - nodes in relevant cat api
BytesStreamOutput default size should be 2k instead of 32k
Recovery Settings : Change settings ( still support old settings ) and allow for more dynamic settings , closes .
Removing request parameters in _analyze API
TEST : Add bwc recovery tests with synced - flush index
REST high - level client : add get index API ( )
Updated docs links to point to master branches
Moved Grok helper code to a separate Gradle module and
Scripting : Groundwork for caching script results ( )
[ ML ] Fix gaps in reserved roles tests ( )
cleanup static member
Add TestWithDependenciesPlugin to build ( )
Fix get - bwc - version for 2 . x releases
[ DOCS ] Document ` doc_as_upsert ` does not support ingest pipelines ( ) ( )
Merge pull request from cbuescher / remove - RescoreParseElement
Introduced ok & acknowledged constant in AcknowledgedRestResponseActionListener
Add TimestampedTable subclass .
Modernize boxplot ' s parser ( )
Fixed bad asciidoc
Merge pull request from javanna / enhancement / gradle_2 . 13
[ TEST ] Raise shardSize parameter if number of shards is > 5
add 0 . 18 . 4 version
[ DOCS ] Fix Fleet links ( ) ( )
Fix testIncorrectPasswordHashingAlgorithm ( )
Make ILM force merging best effort ( )
add fixed thread pool ( mainly for internal testing for now )
[ DOCS ] Fix shard type in CCR overview doc ( )
Fix misspellings .
Add ec2 specific network hosts , closes .
Support Multiple Collapse Inner Hits
Merge pull request elastic / elasticsearch from nik9000 / immutable_map_be_gone
Atomic mapping updates across types ( )
Query DSL : Allow to control how all multi term queries are rewritten , closes .
Tests : Fix RemoteScrollableHitSourceTests
Mapping : Allow to have for a field to not be stored and not indexed , but still be included in ` all ` , closes .
release 0 . 13
[ TEST ] Change ` DatafeedJobsIT ` test to use the external cluster as it doesn ' t start / stop nodes .
Move IndexShard # getWritingBytes ( ) under InternalEngine ( )
Added log line to better debug where rivers will get allocated
[ Test ] Remove unused imports
Docs : Documented index prioritization
move instanceof to catch block
[ DOCS ] [ Security ] Documentation for Role Mapping API ( elastic / x - pack - elasticsearch )
[ GEO ] Add WKT Support to GeoBoundingBoxQueryBuilder
create ` es - 1 . 5 ` branch
Merge pull request from nik9000 / prompts_2
Merge remote - tracking branch ' origin / master ' into feature / client_aggs_parsing
Expand AbstractSerializingTestCase and AbstractWireSerializingTestCase to test diff serialization
SQL : Replace ListTable transport with SQL command ( elastic / x - pack - elasticsearch )
Merge pull request from jpountz / plugin / mapper_size
Merge pull request from rjernst / remove / mapper - close
Fold IndexCacheModule into IndexModule
ilm test - allow more time for policy completion ( )
[ DOCS ] Replace cross - cluster search PNG images with SVGs ( )
[ DOCS ] Review getting started ( elastic / x - pack - elasticsearch )
[ ML ] adds feature_importance_baseline object to model metadata ( )
ExecutionService should be able to process multiple index requests at a time .
Fix some Jenkins build issues ( elastic / elasticsearch )
Remove 7 . 0 . 2 ( )
EQL : Add implicit ordering on timestamp ( )
Short Curcuit response if no indices exits and make sure listener is notified .
[ DOCS ] Document index alias swaps are atomic ( )
Docs : Add Oracle doc link to getting started page
Query DSL : Query String _missing_ and _exists_ syntax , closes .
Docs : Typo
lucene4 : fixed FloatFieldDataTests
[ Docs ] Adding suggestion sections to high level client docs ( )
Fix PMD warning : don ' t reassign method parameters .
Do not log warn shard not - available exception in replication ( )
Add asserting searcher to track searcher references
clean up for methods in SearchRequestBuilder so it works with SearchSourceBuilder
[ ML ] allow forecast only on post 6 . 1 jobs ( elastic / x - pack - elasticsearch )
prepare for next development iteration
Ensure query resources are fetched asynchronously during rewrite ( )
Docs : Warn about the fact that min_doc_count = 0 might return terms that only belong to different types .
Add transport ssl enabled value back to security usage ( elastic / x - pack - elasticsearch )
move things around a bit . . .
Plugins : Reduce complexity of plugin cli
Set network receive predictor size to 32kb ( )
Move test fips configuration to script plugin ( )
Raise maximum index name length to 255 bytes
Remove unused typeParsers method from ParserContext ( )
[ build ] mark elasticsearch as provided in plugins
Snapshot tool : S3 orphaned files cleanup ( )
JDBC : Fix stackoverflow on getObject and timestamp conversion ( )
Improve reporting of assert in IndexShard
Log JVM arguments on startup
fix error when trying to delete all indices and none exists
SQL : Fix checkstyle in sql / rule package ( elastic / x - pack - elasticsearch )
using node settings to get the node name for thread names
[ ML ] Make GetJobStats work with arbitrary wildcards and groups ( )
Use reproducible method of generating properties file for better caching ( )
update documentation with release 2 . 5 . 0
Moved condition errors to their result
[ Docs ] " The the " is a great band , but . . . ( )
Internal : Remove SecretService
Update plugins . asciidoc
Use newIndexSearcher ( ) instead of newSearcher ( ) ( )
Merge pull request elastic / elasticsearch from rjernst / clean_more
[ ML ] Fix native process threading protection and restructure ml threadpools
Build : Log a warning if disabling reindex - from - old ( )
CONSOLE is the new AUTOSENSE
[ TEST ] Remove global scope mentions
Dump Stacktrace on Slow IO - Thread Operations ( )
Disable BWC for backport of as
Add first High level client documentation
Document how to run a single docs test
Restore the original ThreadContext after a preserved context is restored
Build : Set xpack to require keystore
cleanup shield ' s ` Privilege ` and ` Permission ` constructs
Updated outdated default setting in doc .
add named writable registry glue
Switch build system to Gradle
convert max_merge_at_once
add guava as a test dependency for jimfs
[ Docs ] Update aggregations . asciidoc ( )
Docs : Remove the ' Factor ' paragraph to reflect
Handle dynamic versions in func tests assertions
Fix tests
Docs : Improve the exists / missing filters documentation
Convert curl examples to Sense for snapshot restore
Added test with missing type to get
[ ML ] Explain data frame analytics API ( )
SQL : fix * SecurityIT tests by covering edge case scenarios when audit file rolls over at midnight ( )
fix ignored exception
Enable reading auto - follow patterns from x - content ( )
Security : Load system key at startup only
Added trace log statement , to catch stacktraces
HLRC : Add put stored script support to high - level rest client ( )
Migration stats and extended stats to NamedWriteable
Fix migrate tool integration test
add filter and global support
Integration tests for benchmark API .
Update the how - to section of the docs for 7 . 0 : ( )
Forbid Files . isHidden .
Percolate tests
Add cancellation doc for multi search ( )
Make Fuzziness reject illegal values earlier ( )
Merge pull request from nik9000 / line_length_suppress
Fix numeric terms aggregations with includes / excludes and minDocCount = 0 ( )
Bump laster 6 . x version in VersionTests ( )
SQL : Refactor away the cycle between Rowset and Cursor ( )
Update to Elasticsearch 0 . 90 . 6 / Lucene 4 . 5 . 1
[ DOCS ] Adding authorization_realms setting for OIDC ( )
Upgrade to new index settings infrastructure
Get Aliases with wildcard exclusion expression ( )
Added extra logging
Logging : improve logging messages added in &
Add a transformer to translate constant BigDecimal to double
Bump bundled JDK to JDK 13 . 0 . 1 + 9 ( )
Add rest of index stats to cat / nodes and cat / shards .
Address initial review comments
Test : wait for task to start before waiting for it to finish ( )
Renames lifecycle to policy in rest - api - spec ( )
Add MultiSearchTemplate support to High Level Rest client ( )
Upgrade Lucene 8 . 5 to latest snapshot ( )
Reindex remote version lookup test refactor ( )
Remove alpha / beta / rc from version constants ( )
Tests : Remove platform specific assertion in NioSocketChannelTests
Add DateMath support to ScheduledTriggerEvent
Mark IndexWithShadowReplicasIT as AwaitsFix
Remove extraneous references to ' tokenized ' in the mapper code . ( )
[ DOCS ] Add reindex steps for internal indices ( )
Merge pull request from javanna / enhancement / boostable_query_builder_pkg
Always set version , even for x - plugins
[ BUILD ] Fail release if test have AwaitsFix annotations
remove unused variable
Merge pull request from s1monw / issues / 15754
prepare release elasticsearch - analysis - smartcn - 1 . 5 . 0
Fix cat allocation test line length violations
Merge pull request from rjernst / transport_client_deguice
[ Monitoring ] Disable Monitoring Collection by Default ( elastic / x - pack - elasticsearch )
Tests : Various watcher test improvements
Deprecate / _xpack / monitoring / * in favor of / _monitoring / * ( )
Make AnnotationPersister use bulk requests instead of indexing individual documents ( )
[ ML ] Fix typo : decending - > descending
Introduce retention lease expiration ( )
lucene 4 : Upgraded o . e . search . dfs package . ( Distributed idf )
Make AnalyticsProcessManager class more robust ( )
[ TEST ] added more information to fail message for a test to debug test failure
Upgrade to Netty 4 . 1 . 8 ( )
Fix TimeZoneRounding # nextRoundingValue for hour , minute and second units
[ Test ] Mute MonitoringBulkTests
improve logger output on startup
Convert InternalAggTestCase to AbstractNamedWriteableTestCase ( )
Add docs for error file configuration ( )
Move master to Java 1 . 7
test : update active directory certificate
[ ML ] enable logging for test failures ( )
fix LocalExporter minimum version checks with non snapshot versions
Watcher : Remove locking of watches for write operations elastic / elasticsearch ( elastic / elasticsearch )
Doc fix explaining resynchronization with the Cancel command .
[ DOCS ] fix documentation for selecting algorithm for percentiles agg
Allow search to continue when sort field is missing from type mapping , closes .
Document new waiting on shards on index close ( )
Adding support for accessing EC2 and S3 over an http proxy . This is configured using the network . proxyHost and network . proxyPort settings .
inner hits : Query and top level inner hit definitions shouldn ' t overwrite each other .
[ DOCS ] : Indexed scripts / templates
Data Streams : fix bwc test ( )
Add another content - type header in packaging tests
[ BUILD ] exclude target dir from tab validation
add version 6 . 4 . 4
[ DOCS ] Copies ESMS monitoring details to Elasticsearch Reference ( )
Migrate to max line length of 100
Wrap xcontent parser creation in try - with - resource statement where possible
Lazy evaluate java9home ( )
Added path . home setting to BaseQueryTestCase setup , this seems to be enforeced now
Require that the dependent variable column has at most 2 distinct values in classfication analysis . ( )
Cleanup : Removed leftover and unused ip_filter . yml
Move getProperty method out of MultiBucketsAggregation . Bucket interface ( )
introduced AlertContext
Add note about how the body is referenced ( )
Upgrade to Lucene 5 . 5 . 0 official release
SQL : drop BASE TABLE type in favour for just TABLE ( )
Add support for retrieving fields in bulk updates
optional - > provided
[ test ] print test start and end of test setup , execution and cleanup
[ DOCS ] Enabled code snippet testing for datafeed APIs ( elastic / x - pack - elasticsearch )
increase search stats rounds to make sure the stats are higher than 0
Fix failed dependency resolution with external build - tools users ( )
Ensure to release async search iterator in tests
[ Docs ] Fixes a bug in the rest - client watcher docs related to
Changed es version from string to class Version
Remove line - length violations in SmokeTestClientIT
Add setFactory permission to GceDiscoveryPlugin
Netty : Make sure channel closing never happens on i / o thread
[ Rest ] Add the cluster name to the " / " endpoint
Use Java 11 collections conveniences everywhere ( )
Runtime fields : rename emitValue function to emit ( )
Upgrade to lucene - 7 . 0 . 0 - snapshot - ad2cb77 . ( )
Fix minimum compatible version after 6 . 8
Weaken overspecified tests in response snippets
[ ML ] BWC tests for job_stats . timing_stats field ( )
Improve exception handling in transport local execution
Cluster Health API : Add ` wait_for_nodes ` ( accepts " N " , " < N " , " > N " , " < = N " , and " > = N " ) , closes .
Aggregations : reduce histogram buckets on the fly using a priority queue .
Discovery : wait on incoming joins before electing local node as master
Fixed test after merging master
Fix a NPE bug caused by code keep executing after listener returns ( )
Stop using basic license in put license test
Rebooted master - ineligibles should not bootstrap ( )
Abort non - fully consumed S3 input stream ( )
Remove unnecessary boxing and use of deprecated Double ctors
Upgrade beats templates that we use for bwc testing . ( )
[ DOCS ] Enable EQL on docs integ tests ( )
Tests : Revamp static bwc test framework to use dangling indexes
Always log shard ID in ShardStateAction
Enables the ability to inject serialized json fields into root of document . ( )
Ignore project tmp dir .
improve logging when trying to delete unallocated shard , check first if deleting is required , and only then delete it
Fixes issue with dangling index being deleted instead of re - imported ( )
Remove line length violations in o . e . indices package ( )
required changes after merging in upstream branch
Add stale shard issue to Resiliency page
Modify nested classes in ShardStateAction to be static
[ DOCS ] Adds security privilege info to inference bucket aggregation ( )
Added version 1 . 0 . 0 - Beta2
Indexed Scripts : Change preference and thread option for GetRequest .
set line lengths for comments equal to code ( )
Remove PROTOTYPE from children aggregation
[ TEST ] Make sure that " assertNoFailures ( searchResponse ) ; " is used instead of " assertThat ( searchResponse . getFailedShards ( ) , equalTo ( 0 ) ) ; " to see what is failing .
Make SLM Run Snapshot Deletes in Parallel ( )
fix define deprecations of API ' s as a whole and urls ( )
Check term earlier in publication process ( )
Move data stream yaml tests to xpack plugin module . ( )
[ TEST ] Fix store tests to use new ShardRouting APIs
Added the API definition for indexing a document
Use PEM files for PkiOptionalClientAuthTests ( )
Refactor AliasOrIndex abstraction . ( )
Fix GCS Mock Behavior for Missing Bucket ( )
SQL : refactor In predicate moving it to QL project ( )
Switch monitoring to new style Requests ( )
Add ability to get snapshot status for running snapshots
[ CCR ] Change AutofollowCoordinator to use wait_for_metadata_version ( )
Avoid NullPointerException if optional Groovy jar is removed
Fix ignore_failure behavior in _simulate ? verbose ( )
remove register and remove license from license manager service
Adds aggregation profiling to the profile API
Indexed Scripts / Templates : cleanup
[ DOCS ] Add user setup to X - Pack install info ( elastic / x - pack - elasticsearch )
Only fsync global checkpoint if needed
BlendedTermQuery should ignore fields that don ' t exists in the index ( )
upgrade to latest compiled jsr166 libs
Upgrade Tika to 1 . 7
Cleanup QueryTest framework and be more strict
Fixed small bug . Index name should be used to lookup entry .
Moved job lifecycle over to persistent tasks
Adds bucketOrd back to cardinality algorithms ( )
Ensure test replicas have valid recovery state
Fix indentation mistake .
Fix testRunStateChangePolicyWithAsyncActionNextStep race condition ( )
Merge pull request from jaymode / rest_client_ssl
Terms aggregation should remap global ordinal buckets when a sub - aggregator is used to sort the terms ( )
Analysis : fix ignoring tokenizer settings in SynonymTokenFilterFactory
HLRC GraphClient and associated tests ( )
Document proxy flags explicitly and nuke Windows section
Painless : Fix def invoked qualified method refs ( )
Merge pull request from uschindler / painless_java9StringConcats
Upgrade to Lucene 4 . 2 . 1
[ ML ] Change close job api to no longer extend from TransportJobTaskAction .
Handle RejectedExecution gracefully in TransportService
Logging shard level multi percolate errors under debug rather than trace
don ' t spawn on received cluster state
Add note that coerce and ignore_malformed are deprecated for geo
[ DOCS ] Build Elasticsearch Reference from elasticsearch repo ( )
Deprecate types in termvector and mtermvector requests . ( )
Bulk indexing : Fix 8125 hanged request when auto create index is off .
[ DOCS ] Refactored index - templates topic . ( )
correct expected thrown exception in mappingMetaData to ElasticsearchParseException ( )
[ DOCS ] Update upgrade docs for 7 . 7 ( )
Ingest Attachment : Upgrade Tika to 1 . 18 ( )
Basic scroll tests
Update for clarification
Correct outdated information in _index docs . ( )
[ test ] generate unique user names ( )
[ DOCS ] Reformat cat fielddata API ( )
[ TEST ] don ' t cache store stats in tests that ensure stats are correct
Remove the _all metadata field ( )
Added multi percolate api
Make the run task honor tests . es properties ( )
[ TEST ] Improve diagnostics for ML interim results test failure
Fixes shrunk shards allocation check step
Whitelist node stats indices level parameter
Document standard metadata entries . ( )
Remove unused interface InitialStateDiscoveryListener ( )
Add Netty channel information on write and flush failure
moving version to 2 . 0 . 0 - SNAPSHOT
Merge pull request from rmuir / enable_j9
fix javadoc usage
Fix wrong property name ( )
[ test ] change id between tests , to make reading log files easier .
Acknowledge problem for put mapping with multiple indices or all indices , closes .
[ TEST ] resolved compiler warnings in ShapeBuilderTests and GeoPointParsingTests
Added zentity to the list of API extension plugins ( )
simplify flow / stop control in testConcurrentWriteViewsAndSnapshot
Don ' t mkdir directly in deb init script
Build : Split packages into submodules
Remove deprecated endpoints containing _xpack . ( )
[ DOCS ] Added caveat about the number of file realms ( )
_default_ mapping not applied when using separate master / data nodes
- removed the getKeyAsString method from StringTerms ( redundant method )
no need for thread pool to notify listeners
[ DOCS ] Minor typo fixes in reindex . asciidoc ( )
S3 Repository : Remove bucket auto create ( )
Tests : Fixing ChainIntegrationTest to work without timewarp as well
Mute testToQuery test
test : ensure address is resolvable in CertUtilsTests # testSubjectAlternativeNames
Decouple BulkProcessor from client implementation ( )
[ ML ] fix inference ml - stats - write alias creation ( )
Unmuted testAutoFollowManyIndices ( ) test and added more logging ,
Compute declared versions in a static block
Add xpack . eql . enabled feature flag , disabled by default . Enabled only for integration tests . ( )
EQL : Enable additional logging in RestEqlCancellationIT ( )
add missing license header
use the same improved fields extraction in get api
Remove explicit parallel new GC flag
rename prefix_length to prefix_len for consistency
Initial commit
Turn off real - mem breaker in REST tests
[ Test ] Fix AggregationsTests # testFromXContentWithRandomFields
Calculate artifact checksums in maven
Docs : Fix rendering of APIs that mention version ( )
Update deprecation logging doc with logger configuration ( )
Move JdbcTemplate
Make testcase more reproducible
Fix ingest simulate verbose on failure with conditional ( )
Snapshot / Restore : fix FSRepository location configuration
Move RestGetInfluencersAction under results package
[ DOCS ] Rewrite analysis intro ( )
[ SPEC ] Added indices . exists_template . json
Muting AsyncSearchActionIT . testTermsAggregation
Merge remote - tracking branch ' es / master ' into ccr
lucene 4 : change Unicode utils to use BytesRef instead of UTF8Result
Add release notes to docs index
[ JAVA7 Upgrade ] Replace explicit type with < >
Remove ParseFieldMatcher usage from AllocationCommands
Fix third - party audit tasks on JDK 8
continue abstracting json into xcontent
Triggering an update
Add a Painless Context REST API ( )
Clarify development vs . production mode
close stream if file not found ( though a stream is not really opened in this case . . . )
Read git revision directly from repository ( )
Remove type - related methods from QueryBuilders ( )
[ DOCS ] Note that EQL timestamp field can also be date_nanos
drop binary mapping BWC test
Release : Add release script for S3 uploading
Merge pull request from talevy / no_match_for_grok
[ TEST ] fixed typo in _cat / thread_pool docs
Docs : make the welcome page more obvious
Remove stale comment from JVM stats ( )
Docs : add documentation about removal of cluster . routing . allocation . snapshot . relocation_enabled
Some small cleanups around the lookup
Node Stats : Add number of server open channels for transport and http , closes .
Added dedicated test for MultiMatchQuery
Mute SSL HTTP connection test
Mute testTracerLog
Added foreach in grammar .
Add missing license check files .
Docs : Fixed bad links
Fix overlapping getters in Function vs AggFunction
Move to use serial merge schedule by default
EQL : Replace EqlSearchResponse . Hits parser with ObjectParser ( )
Adapt version after backport of ( )
Fix the building of the default URL for the setup password tool ( elastic / x - pack - elasticsearch )
Merge pull request from jasontedor / maps - be - gone
SQL : Check case where the pivot limit is reached ( )
SQL : Fix Nullability of DATEADD ( )
Allow to search across indices and types even if some types do not exists on some indices , closes .
Merge pull request from MaineC / deprecation / script - sort - builder
[ Test ] Add full cluster restart test for Rollup ( )
when serializing mappers into json , also serialize index , type and id mappings ( index for the enabled flag , and id / type for the store )
Field Cache : Allow to set ` index . cache . field . expire ` when using ` resident ` type , closes .
fix : Sort Processor does not have proper behavior with targetField ( )
Factor out PID file creation and add tests
Allow AVX - 512 on JDK 11 + ( )
Plugins : Remove guice from Mapper plugins
only creates initialisation service ifeature is enabled
Internal : pending operations in the translog prevent shard from being marked as inactive
Fix handling of document failure expcetion in InternalEngine ( )
muting test ( )
Only ignore gradle wrapper in root dir ( )
Document strict dynamic type mapping .
Bump version in BWC check after backport
clarify comments and add assert client auth type
CONSOLE - ify date_range aggregation docs
[ TEST ] Don ' t start thread before checking for pending refresh
Allow cluster access during node restart ( )
[ DOCS ] Add data streams to API conventions ( )
[ Docs ] Fix section title in profile . asciidoc
Add ILM package info documentation ( )
Testclusters : start using it for testing some plugins ( )
[ 8 . 0 ] Version bump 7 . 9 . 1 release
make every static field final ( immutable )
Handle special chars in JAVA_HOME in elasticsearch - service . bat ( )
Docs : Added 5 . 0 . 0 - alpha1 release notes
[ FIX ] call onDisabled at first if appropriate
Decode URL . getPath before resolving a real file
Change SearchTypes to be ` QUERY_THEN_FETCH `
Revert " Security : don ' t call prepare index for reads ( ) "
migrate branch for analysis - stempel
[ DOCS ] Fix default setting in circuit breaker documentation
Group client projects under : client
Test : muted testStaleMasterNotHijackingMajority
cleanup
ZenDiscoveryUnitTests should close objects in reverse order of creation
Add precise logging on unknown or invalid settings
Prevent deletion of ILM history template in rest tests ( )
Unmute field collapsing rest tests
Add ` allowed_warnings ` to yaml tests ( )
Introduce on_failure_pipeline ingest metadata inside on_failure block ( )
Fix BlobStoreIncrementalityIT ( )
update to next 0 . 16 . 0 version
Don ' t ignore ` ignore_unmapped ` options on nested , has_child and has_parent queries if the wrapped query gets rewritten .
Docs : Reenable rollup docs tests ( )
Remove Index Reader warmer introduced in 0 . 90 . 6
[ DOCS ] Adds anchors and x - pack icons for frozen indices ( )
[ DOCS ] Fix link to quartz crontrigger tutorial ( )
Fix illegal cast of the " low cardinality " optimization of the ` terms ` aggregation . ( )
Build : Fix artifact id for xpack transport client
Warn on not enough masters during election ( )
unknown run as users should not be allowed to execute any APIs
HLRC support for Index Templates V2 ( )
Fix markup
CCR : use shadow plugin
Stricter checks of setup and teardown in docs tests ( )
[ DOCS ] Removed X - Pack Breaking Changes
Unify transport settings naming ( )
Remove and forbid the use of com . google . common . base . Predicate ( s ) ?
[ Tests ] add different node name prefix for the different cluster type
Clear the GroovyClassLoader cache before compiling
[ DOCS ] Fix 404s ( )
[ CCR ] Validate follower index historyUUIDs ( )
add missing import
[ Tests ] NoMasterNodeTests make timeout checks less sensitive
Make compilation pass .
Unassign DFA tasks in SetUpgradeModeAction ( )
Deprecate lat_lon and precision_step
Test : Temporarily change delete / put_mapping to wait for green
Put JVM crash logs in the default log directory ( )
[ mappings ] update dynamic fields in mapping on master even if parsing fails for the rest of doc
fix no = > not typo ( )
Upgrade to Lucene - 7 . 5 . 0 - snapshot - 608f0277b0 ( )
Logging : shard store shouldn ' t log the entire cluster state under trace , but just its version .
Fix some typos ( )
Adjust bwc assertion after backporting
Forbid equality checks against ' true ' ( )
Fix sporadic SimpleQueryStringBuilderTests failures .
New aggregations feature - “ PercentageScore ” heuristic for significant_terms aggregation provides simple “ per - capita ” type measures .
Search - make term / prefix / wildcard / regex query parsing more lenient ( )
ShieldIndexSearcherWrapper should create the scorer only once . elastic / elasticsearch
Query DSL : ` query_string ` - Expose QueryParser # setAutoGeneratePhraseQueries , closes .
Remove Strings # splitStringToArray
SQL : Fix failing test due to random timezone ( )
Cut filters aggregation to registerAggregation
Fix misspelling .
Add soft limit on allowed number of script fields in request ( )
Fixes bug preventing script sort working on top_hits aggregation ( )
Merge test runner task into RestIntegTest ( )
[ DOCS ] Fix typo in search profile docs ( ) ( )
Checkstyle fix
Remove support for field names in node_stats url
Mute second test in monitoring / bulk / 10_basic
add ` ignore_missing ` option to SplitProcessor ( )
upgrade to mvel 2 . 1 Beta6
changed the actions format and added the notion of an action id
Run sort optimization when from + size > 0 ( )
Add copy constructor to SearchRequest ( )
[ DOCS ] Reworks some parts of EMM API docs ( )
Verify consistency of version and source in disruption tests ( )
Fix missing break statement causing reroute serialization failure
Merge remote - tracking branch ' origin / master '
switch to plugin pom
Comment on the extra reroute after failing shards ( )
Require JDK 9 for compilation ( )
Fix sorting on nested field with unmapped ( )
Merge remote - tracking branch ' dakrone / disable - all - by - default '
Merge pull request from brwe / log - config - order
test : Make many percolator integration tests real integration tests
Fix potential NPE in FuzzyTermsEnum ( )
Build : Fix deb version to use tilde with prerelease versions ( )
Added an AwaitsFix to testDeleteAllOneIndex . Martijn is on it .
Update wire protocol after backport ( )
if ttl field mappers don ' t exists ( yet ) , ignore it for that shard
Relax elapsed time stats assertion ( )
Prevent Leaking Search Tasks on Exceptions in FetchSearchPhase and DfsQueryPhase ( )
upgrade to rhino 1 . 7R4
[ DOCS ] Added ML calendar APIs ( elastic / x - pack - elasticsearch )
[ DOCS ] Add anchor to version types list . ( )
Tribe : Passthrough environment and network settings to tribe client nodes
Use nio driven http client for testing ( )
Support unknown fields in ingest pipeline map configuration ( )
Add indexing pressure documentation ( )
Core : Fix epoch millis java time formatter ( )
Add CONSOLE annotation to sort documentation
Add tera and peta to RestTable . renderValue ( )
allow to set a query using a string as well
Don ' t schedule SLM jobs when services have been stopped ( )
A new ` _shard_stores ` API provides store information for shard copies of indices .
Make sure 2 writes happened , so we can check if 2 deletes happen that have been triggered by ttl
[ TEST ] Pass class level test logging to external nodes
Remove doc links from config template
Test : Let both types have a non_analyzed id field .
SQL : Improve serialization of SQL processors ( )
Elasticsearch support to JSON logging ( )
Re - enable data stream bwc tests ( )
SearchableSnapshotDirectory should not evict cache files when closed ( )
Use routingResult method to update cluster state after reroute
wrap line after 140 chars
cleanup unused term factory
Simplify TcpTransport interface by reducing send code to a single send method ( )
Preserve response headers when creating an index ( )
Merge pull request from jpountz / fix / highlighter_extract
Adjust validation endpoints ( elastic / elasticsearch )
[ DOCS ] Changes level offset of anomaly detection pages ( )
[ DOCS ] Reformat truncate token filter docs ( )
Third party audit improovements ( )
[ ML ] Create state index and alias before starting an analytics job ( )
Fix calculation of time of term vectors request
Fix doc link for HLRC async Put Lifecycle API ( )
Revert " Add global checkpoint assertion in index shard "
Use a unique GCE_HOST setting
Change JDK distribution source ( )
use parsing field to check for order
Add timeout for ccr recovery action ( )
release 0 . 90 . 0 . RC1
Aggregations enhancement - remove pointless term frequency lookups .
Adressing review comments
add indices flag to nodes info request ; use boolean to express ' null ' indexing buffer value on the wire
simpler settings builder api
Stop map collisions on FiltersTests
Validate known global settings on startup
[ TEST ] Made sure that the parent exists ( ` foo ` is the parent type and not ` parent ` )
When stopping alert manager wait for ongoing operations to complete .
break field data value iterator into two callback types , one with support for missing , and one without
Move old jdk download tests to openjdk tests ( )
fix test for existance of directory based mappings when creating an index
malformed elasticsearch . yml causes unresponsive hang
[ TEST ] simplify number of data nodes in # testReplicaCorruption
add a proper master not discovered exception for master based operations ( cluster state ) and the ability to control the timeout till master is found
Merge remote - tracking branch ' elastic / master ' into feature / sql
Revert " HLRC : Add ILM Status to HLRC "
[ TEST ] Mock getVersion on channel for ServerTransportFilterTests
Allow to pass es . default . settings to the process , using it as default value unless specified in the config file , closes .
Add information about in - flight requests when checking IndexShard operation counter ( )
Update shade - plugin to 2 . 3
Added a link to official Python client to the client list , fixed perl link
Simplify adding plugins and modules to testclusters ( )
Push runtime Java home through to Gradle
Change value of METADATA_REFRESH for the tests ( elastic / x - pack - elasticsearch )
add debug logging of the ping responses received
Query Refactoring : remove deprecated methods and temporary classes
Merge pull request from sarwarbhuiyan / mapping - spelling - error
Remove most usages of 1 - arg Script ctor ( )
Docs : Rewrote the upgrade section
Add support for http input endpoints that do not return JSON formatted bodys .
[ ML ] Move consuming and closing results stream to result processor ( )
Roll translog generation on primary promotion
Remove extra bin / directory in bin folder
Docs : clarify cluster state handling
Filter Cache : Improved Caching , closes .
Update index - modules . asciidoc
Open correct ( renamed ) index on restore
Promote common rest test utility methods to ESRestTestCase
Merge pull request from rjernst / remove / mapper - generic
remove some safe methods because of the new makeSafe method usage
Only add unicast provider if all settings are present
Update number of replicas when removing setting ( )
[ Test ] Additional logging for RemoteClusterClientTests ( )
More unit tests
fix another test
on failure to extract a plugin , log the message and bail
Core : Change the rest api url structure as if all operations are targetted for the ` . alerts ` index .
Fix version number for elasticsearch < 1 . 3 . 5
Use startNodesAsync in more tests
Fix test in mapper attachments plugin
Merge pull request from jpountz / fix / ungroovy_AliasesRouting + UpdateIT
[ ML ] Align aggregated data extraction to histogram interval ( elastic / x - pack - elasticsearch )
[ DOCS ] Updates terms in machine learning datafeed APIs ( )
Remove unnecessary imports
License Plugin : Removed installing a special license in the delete REST test
[ DOCS ] EQL : Document multi - value field support ( )
fixed a bug in PrioritizedThreadPoolExecutor :
Merge pull request from martijnvg / parent - child / bug / _parent_field_validation
Merge pull request from nik9000 / config_tests
Splits nio project into two for eclipse build only ( )
Remove UnknownNamedObjectException ( )
SQL : change the default precision for CURRENT_TIMESTAMP function ( )
[ TEST ] restore tests repeatability in AbstractQueryTestCase
improve handling of translog snapshotting
remove sneaked placeholder from histogram docs ( )
REST high - level client : add support for Rollover Index API ( )
[ DOCS ] Added index - shared4 and index - shared5 . asciidoc
Add sync_id to cat shards API
Move netty transport and http into a module
Removes parent child fielddata specialization ( )
Mute ReplicationTrackerRetentionLeaseTests # testAddOrRenewRetentionLease ( )
Add notice about official Rust client ( )
fs stats should be public
Add QA project and fixture based test for discovery - ec2 plugin ( )
Refactor actions to match trigger refactoring .
[ Internal ] user node ' s cluster name as a default for an incoming cluster state who misses it
Update settings : Allow to dynamically update thread pool settings
Remove Unused AckClusterStatePublishResponseHandler ( )
Make xpack extensible :
Add note how to run locale sensitive unit test ( )
add missing ' of '
[ DOCS ] Updated release state for 7 . 0 . 0 - alpha1 ( )
Put CCR tasks on ( data & & remote cluster clients ) ( )
Merge pull request from LeeDr / fixStoredFieldsMessage
Shared Gateway : Allow to dynamically update the ` snapshot_interval ` using update settings , closes .
Merge pull request from rmuir / rr_upgrade
add a hack to see if this fixes windows issues
[ DOCS ] Collapse nested objects in node stats API response ( )
Improve painless docs for score , similarity , weight and sort ( )
No need to find replica copy when index is created
Adjust the version for 250_distance_feature test
Remove usage of deprecated compare gradle builds plugin ( )
Reuse existing allocation id for primary shard allocation
release 0 . 7 . 0
Use project - defined Java installation for keytool ( elastic / elasticsearch )
prepare for next development iteration
added priority support for cluster state updates :
percolator : Add support dor MatchNoDocsQuery in query terms extract service
Revert " Doc : / _reindex : Add a note about the source size parameter "
Remove netty_3 support from xpack ( elastic / elasticsearch )
Support externalValue ( ) in mappers
Suppress Wildfly test in FIPS JVMs ( )
Merge pull request elastic / elasticsearch from rjernst / spec_in_resources
First steps at whitelist cleanup
[ ML ] Improve hard_limit audit message ( )
Use InputStreamStreamInputs validation to limit size of tokens ( elastic / x - pack - elasticsearch )
Fail Snapshot on Corrupted Metadata Blob ( )
Always Accumulate Transport Exceptions ( )
Disable java and maven version checking
Fall back to Java 13 APIs for forbidden API checks when using JDK 14 ( )
Start building analysis - common module ( )
Docs : Removed old coming / added tags
Skip REST YAML tests in FIPS 140 mode ( )
add value_script as an option to terms stats
[ DOCS ] Add response body parms to search API docs ( )
Add note to keystore docks on requirement for restart to take effect ( )
Scripting : Execute Scripting Engine before searching for inner templates in template query
Fix license expiry log message
[ TEST ] make sure all shards have docs
Remove tcp profile from low level nio channel ( )
[ DOCS ] Drafted overview of calendars and special events ( elastic / x - pack - elasticsearch )
upgrade to latest assembly
Move TupleMatchers class to org . elasticsearch . test . hamcrest package ( )
Fixing typo with default throttle period
Consistent case in CLI option descriptions ( )
Merge pull request from jasontedor / fix / 12683
Add these two props for consistency , so we aren ' t confused
Merge pull request : Add shuffling xContent to aggregation tests
Tests : Improve rest suite names and generated test names for docs tests
Update to elasticsearch 1 . 4 . 0
Stop automatically nesting mappings in index creation requests . ( )
Add InboundHandler for inbound message handling ( )
Build ElasticSearch as Debian package
cutover jarhell to parsing classpath all the time
Fixed issue where the parentTypes set would be updated when a new parent type is being added or removed during a refresh , which would have lead to concurrency issues .
Scripting : Move context definitions to instance type classes ( )
Testing : Fix ClusterPrivilegeTests by using jsonBuilder instead of handcrafting JSON
Use ` _language ` field instead of ` language `
[ TEST ] Add a test that ensures Version # CURRENT is actually the latest version
Disallow { index } being specified at the top level of _xpack / monitoring / _bulk
Add rest put mapping test with dot in fieldname
Routing : Restore shard routing .
Docs : Add Upgrade API documentation ( elastic / x - pack - elasticsearch )
nicer code
[ Discovery ] Fault detection should also check cause exceptions for disconnects
Docs : Use " js " instead of " json " and " sh " instead of " shell " for source highlighting
allow to set settings on the groovy client
MapperService : check index . mapper . dynamic during index creation
Docs : Added redirect entries for multicast plugin and the cloud plugins
Query refactoring : SpanNearQueryBuilder and Parser
Tweak Vagrant to use env variables for S3 boxes , update instructions
Update client benchmarks to log4j2
Fix method signature in test
Refactor script context generic types ( elastic / x - pack - elasticsearch )
[ DOCS ] Role Mapping API improvements ( elastic / x - pack - elasticsearch )
Remove the node local storage setting ( )
[ TEST ] Added awaits fix to failing LoggingConfigurationTests
remove grok and join usage from IngestClientIT
Merge pull request from rmuir / systemassertions
Merge pull request elastic / elasticsearch from s1monw / new_index_settings
Allow skipping ranges of versions ( )
add profile name to TransportChannel
Cloud Plugin : Cloud gateway default chunk size change to 1g , closes .
Ensure no circular reference in translog tragic exception ( )
Do not cache the QueryShardContext in PercolatorFieldMapper : it is cheap to create .
[ DOCS ] adding a note on python client versioning schema
adapt to upstream changes
test fixes
Move forbidden suppression for InetSocketAddress to its own method
Adjust RestHighLevelClient method modifiers ( )
optimize array list creation with exact size
remove dead code
Fix date format in warning headers
[ Docs ] Fix typo in pattern - tokenizer . asciidoc ( )
Add wait_for_no_initializing_shards to cluster health API ( )
Geo context suggester : Require precision in mapping
Fix ilm / 20_move_to_step basic moving to step ( )
Add FieldMappers before adding Mappers ( used for parsing ) and while holding the relevant ' s ObjectMapper mutex
Convert ` config . ignore_system_properties ` to new settings infra
Add a late - parsing SearchSoruce wrapper for index warmers
Fix handling of exceptions thrown on HEAD requests
Merge pull request from jasontedor / intellij - type - inference
Replace Streamable w / Writeable in BaseTasksRequest and subclasses ( )
Removes version 2 . x constants from Version ( elastic / x - pack - elasticsearch )
Merge pull request elastic / elasticsearch from rjernst / deguice16
Follow - up of , same fix applied to ToXContent . MapParams
Small Simplifications DiskThresholdDecider ( )
mute testAckedIndexing
better logging on failure to create index
Snapshot / Restore : remove obsolete expand_wildcards_open and expand_wildcards_close options
Merge pull request from jpountz / test / rest_ssl
REST test for rolling data streams ( )
Upgrade to netty 3 . 3 . 0
Adds debug logging to RepositoryUpgradabilityIT test to help figure out
Updated test to reflect added acknowledgement from all nodes in open / close index api
Removed custom naming for DISTINCT COUNT ( )
Update CONTRIBUTING . md to indicate Gradle > = 4 . 3 is required ( )
[ STORE ] Use DistributorDirectory only if there are more than one data direcotry
Fix API usage after index - level guice was removed
[ DOCS ] Add code snippet testing for more ML APIs ( )
Merge pull request from jpountz / upgrade / hdr_histogram_216
Allow include / exclude of completion stats via REST parameters
[ ML ] Add missing doc links to df - analytics rest spec and HLRC javadocs ( )
Don ' t Ping on Handshake Connection ( )
Updated the test README to allow multiple test sections and an initial setup section
Fix BlobStoreIncrementalityIT Assertion ( )
[ Transform ] introduce new roles and deprecate old ones ( )
[ Docs ] Minor formatting and wording fixes ( )
Tests : disable rebalancing in restoreIndexWithShardsMissingInLocalGateway
Fix formatting of the documentation .
Tests : Change rest test extension from . yaml to . yml ( )
Nodes Stats API , closes ,
Remove optionality from streaming useCompressedOops field
Fix funny generics in request size limit test
[ DOCS ] Updates screenshots in ML population analysis ( )
Merge pull request from rmuir / bwcompat1
Use Supplier instead of Reflection
Upgrade bundled jdk to 14 . 0 . 1 ( )
[ DOCS ] Move appropriate TLS content to Elasticsearch Ref ( elastic / x - pack - elasticsearch )
[ CCR ] Introduce leader index name & last fetch time stats to stats api response ( )
[ ML ] Allow xpack . ml . max_machine_memory_percent higher than 100 % ( )
also add it as a parameter
test : use the correct stop watcher url
test : fix forbidden api failure
Enable option to use request cache for size > 0
Mappings : Make mapping updates atomic wrt document parsing .
added predefined empty implementation for all atomic field datas
Tests : Make MockWebServer thread - safe ( elastic / elasticsearch )
more work on attachments , have basic types support external set values
Use standard format for reload settings API ( )
Updated links to maven repo to use https . Closes elastic / elasticsearch .
Remove empty javadoc ( )
Revert " AwaitsFix for "
Fixed parsing bug in percolator , where everything after the ` doc ` object was skipped .
store names of indices as an array instead of a string
Deprecate the _termvector endpoint . ( )
Rename DiscoveryNodes # localNodeMaster to isLocalNodeElectedMaster
Docs : Add note about which secure settings are valid ( )
Document that painless doesn ' t support the " advanced text scoring " ( )
Allow for the sort search element to speciy the sort fields in an array , closes .
Remove unused IndicesOptions # fromByte method ( )
[ Doc ] Update plugins intro about categories of plugins core / community
no need ask for the first query rest twice
Moving join ( ) helper function to WordScorer
Create branch es - 1 . 4 for elasticsearch 1 . 4 . 0
Fix implicit duplicate duplicatesStrategy in processResources ( )
[ TEST ] Reenable UnicastZenPingTests . testSimplePings
remove impl details from method name
Honour ignore_unavailable option when resolving indices
Use JDK 10 for 6 . 4 BWC builds ( )
Add a doc value format to binary fields . ( )
Updated fielddata docs to make it easier for users with old mappings
add tests for concurrent user lookup
Fix typo in ScriptDocValues deprecation warnings ( )
Query refactoring : SpanFirstQueryBuilder and Parser
Remove _reload_search_analyzer experimental status ( )
Added a status method CountResponse that resolve shard failures into rest status code . That method is now used in RestCountAction to return proper status .
Rename script field type to runtime_script ( )
Upgrade Azure Storage client to 4 . 0 . 0
Mute failing IndexShardIT # testPendingRefreshWithIntervalChange
Mute test CCRIT testIndexFollowing ( )
Zen2 : Turn to follower on follower check when no state accepted yet from new leader ( )
[ DOCS ] Add verify snapshot repository API docs ( )
Added @ TestLogging to all methods to see which shards we refresh and which ones we search on
[ DOCS ] Removes beta label from index sorting ( )
ignore interruped exception when scheduled refresh is going on
dynamic flag on object type should lower case itself when serialized back to its json representation
Made template endpoint compatible with search endpoint
Remove object notation for core types .
Update token - count . asciidoc
make mult default boost mode
disable content - type randomization for some ml tests
[ DOCS ] Updates security configuration overview ( )
[ TEST ] RecoveryWhileunderLoadTests sometimes need higher timeouts
[ TEST ] Made ttl test more predictable
Create branch es - 1 . 4 for elasticsearch 1 . 4 . 0
rename pipeline_id param to pipeline
Fix NormalizerAgg test searcher wrapping ( )
[ DOCS ] Change standalone node to dedicated node ( elastic / x - pack - elasticsearch )
more fixes for tests
Update geo_point docs for geo_shape queries ( )
Add basic license integ test
Replace Math . toIntExact with toIntBytes ( )
Mappings : Remove ability to disable _source field
Fix handling of invalid error trace parameter
Truncate tlog cli should assign global checkpoint ( )
Update usage of es . staging . plugins sysprop to pass build hash
Merge pull request from shikhar / patch - 1
cleanup redundant schedules array
[ ML ] Improve handling of job errors from datafeed ( elastic / x - pack - elasticsearch )
[ DOCS ] Fix realm setting names ( )
Reenable test in ` indices . put_mapping / 20_mix_typeless_typeful . yml ` . ( )
[ TEST ] wait for no pending tasks so processed all mappings
fix PolicyStatsTests : mutate kept same indicesManaged
[ DOCS ] Remove response params for ( )
Recheck cutoffScore during phrase_suggest merge .
[ Docs ] Emphazise suggest behaviour with missing query part ( )
[ ML ] Get ML filters size should default to 100 ( )
Call validate in TransportClients before request is send
update doc
mark slow tests with @ Slow annotation
Search : Add search_type for ` count ` to return count but still support facets , closes .
Always attach system user to internal actions ( )
Mappings : Remove AbstractFieldMapper
Fix m2e Eclipse Integration
TEST : Mute testEnsureWeReconnect
Increase IO direct byte buffers to 256KB ( )
Fix NPE in SearchSourceBuilder
Rename text query to match query ( text query still works , with variants ) , closes
add master_timeout to cluster state changing operations
Preserve ILM operation mode when creating new lifecycles ( )
Add trace logging to AllocationDeciders if decision is NO ( )
Updated nodes info endpoint in elasticsearch . yml
CharArraySet doesn ' t know how to lookup the original string
Update README with full command to generate spec
allow to disable the optimization of removal of ords on single value numerics / geo field data
use ShardPath . getRootStatePath ; allow forbidden API
Enabled cannot be updated ( )
[ DOCS ] Removes redundant file realm settings ( )
Update . gitignore to ignore the execution hints files
Add YAML REST tests for filters bucket agg ( )
[ TEST ] Move test skip / blacklist assumptions out of @ Before method ( )
Core : surgically removed slow scroll , because master ( 2 . 0 ) requires full cluster restart coming from previous versions .
Merge pull request from dadoonet / master
Check inner field with metadata field name is ok
Refactor FieldSortBuilder
[ ML ] Fix bug initialising ML Metadata ( elastic / x - pack - elasticsearch )
CSV ingest processor ( )
Revert " Add FastVectorHighlighter support for more complex queries . "
[ ML ] Allow model_memory_limit to be reduced ( elastic / x - pack - elasticsearch )
[ ingest ] geo - ip performance improvements ( )
Replace required pipeline with final pipeline ( )
fix reenable spaces in paths during integration tests
Build : Use JDK at JAVA_HOME for compiling / testing , and improve build info output
Use ` addIfAbsent ` instead of checking if an element is contained
[ ML ] Move analyzer dependencies out of categorization config ( )
CORS : Disable by default
Show JVM arguments
Removes FailedRerouteAllocation and StartedRerouteAllocation
Reduce memory pressure when sending large terms queries . ( )
Format settings in discovery - ec2 docs
Added ability to remove pipelines via wildcards ( ) ( )
Add doPrivilege blocks for socket connect ops in repository - hdfs ( )
[ FIELDDATA ] Use KeyedLock in IndexFieldDataService
Add JAVA_HOME env override location to docs ( )
[ ML ] Stop immediately when job isn ' t opened
Merge pull request from mikemccand / remove_merge_docs
Added the ` cluster . put_settings ` API
bulk : Fields defined in the ` _default_ ` mapping of an index template should be picked up when an index alias filter is parsed if a new index is introduced when a document is indexed into an index that doesn ' t exist yet via the bulk api .
on index failure when updating , make sure to call the listener
Test : CorruptedFileTest . testCorruptionOnNetworkLayer used node settings when creating an index
SQL : Fix esType for DATETIME / DATE and INTERVALS ( )
fix spelling and remove wildcard imports
Drop all fully deleted segments for comparison
Merge pull request elastic / elasticsearch from rjernst / warnings_cleanup2
Add Lucene version in plugin . properties file
[ TEST ] Assert that put mappings requests are acked in GeoMappingTests .
Fix log parameters
Completion postings format : Write hole character
XContent - An abstraction on top of content ( JSON inspired ) , closes .
fix targetField randomization in JoinProcessorTests ( )
Support distance units in GeoHashGrid aggregation precision ( )
Test mute for issue 60212 ( )
Add comment explaining priority of JVM options ( )
docs and wrap the rest bulk request in an array
Removed unused nested children collector .
Added unit tests for ClusterIndexHealth and ClusterHealthResponse
[ SPEC ] Removed percolate from index . json
Merge pull request from rmuir / jarhell
Core : Perform write consistency just before writing on the primary shard
Fix tribe node cluster state version increments ( )
Split scroll as separate endpoint
Omit non - masters in ClusterFormationFailureHelper ( )
Fix Typo in Snapshot Abort Test ( )
Completion types with multi - fields support ( )
Pass COMPUTERNAME env var to elasticsearch . bat ( )
Update . ml - config mappings before indexing job , datafeed or df analytics config ( )
Add support for ignore_indices to delete by query
Merge pull request from jpountz / fix / multi_level_breadth_first
Added deleteBlob IOException test
[ ML ] Make ml internal indices hidden ( )
Switch more docs to CONSOLE
Geo : Makes coordinate validator in libs / geo plugable ( )
remove StringValues in favor of BytesValues
Throw LockObtainFailedException exception when we can lock index directory
Only apply initial recovery filter to shrunk shard ( )
prepare for next development iteration
Merge pull request from ywelsch / fix / reuse - alloc - id
Mute EllipticCurveSSLTests
[ TEST ] Reimplement AckTests # testDeleteWarmerNoAcknowledgement
Move to Elasticsearch 0 . 21 . 0 . Beta1
Create branch es - 1 . 4 for elasticsearch 1 . 4 . 0
[ DOCS ] Adds example to the inference aggregation description ( )
Do not apply minimum_should_match on auto generated boolean query if the coordination factor is disabled .
Fix exit code in certutil packaging test ( )
add getter for open file desc
[ DOCS ] match description with example in allocation docs page ( )
Fix failing stylecheck
Remove setting index . optimize_auto_generated_id ( )
Tests : Move static bwc indexes to a shared location
Remove and forbid final uses of ImmutableList
Allow leading ` / ` in AUTOSENSE path
Fix reschedule async fsync test
Merge pull request from areek / docs / clarify_plugins_installation
Fix GCS Mock Broken Handling of some Blobs ( )
Increase default write queue size ( )
Update docs for the DFR similarity ( )
Add version 5 . 4 . 2 constant
[ ML ] Wait for controller to respond to commands ( )
[ ML ] adds multi - class feature importance support ( )
updated testScriptConditionParserBadScript ( ) test to face reality
[ TEST ] Don ' t assert on null value . It ' s fine to not always see an exception in this part of the test .
[ ML ] Refactor ProcessCtrl into Autodetect and Normalizer builders ( )
set missing create param in PutRequest
Switch to NamedWriteable pull based extension in xpack
[ TEST ] Don ' t rely on ` new Date ( ) `
Aggregations : Fix multi - level breadth - first aggregations .
Painless : Modify Loader to Load Classes Directly from Definition ( )
Fix Noisy Logging during Snapshot Delete ( )
add network info printing
cut over more tests to AbstractIntegrationTest
Merge remote - tracking branch ' es / master ' into enrich
[ TEST ] address testCollectNodes rare failure ( )
improve update test to wait for green cluster state
increase logging level to DEBUG in backwards - 5 . 0 tests
Merge pull request from jasontedor / shard - failed - listener
TEST : Fix InternalEngine # testAcquireIndexCommit
Merge pull request from rmuir / fail_if_no_tests
fix wrong parsing of useDisMax with number
Fix client test to not care about complete error response message
Shield Docs : Fixed another cross doc problem .
Improve the way sub - aggregations are collected .
Add tests for FileAttributesChecker
close open index input on failure
prepare release elasticsearch - analysis - smartcn - 1 . 6 . 0
Remove broadcast address check .
Removed delete tests which accidentally got re - added on merge
geoip : don ' t store geoinfo if nothing can be resolved
[ ML ] Be nicer on platforms that don ' t have the ML binaries ( elastic / x - pack - elasticsearch )
start abstaction of json
Prepare for backport of bounds in aggregations refactoring ( )
[ maven ] change also artifactId for x - dev - tools
Assorted fixes for bugs in the PagedBytesReference tests , plus a typo
Remove random of recovery chunk size setting
Update lang - analyzer . asciidoc
[ TEST ] lower to 5s the wait for the Thread . sleep
[ DOCS ] Correct deprecation note in mapping docs ( )
[ HLRC ] Update Stop & Start ILM APIs to use Validatable ( )
Disable console logging in the Windows service
Correctly list blobs in Azure storage to prevent snapshot corruption and do not unnecessarily duplicate Lucene segments in Azure Storage
mute testSlmPolicyAndStats
Merge pull request from nik9000 / standalone_xlint
Un - assign persistent tasks as nodes exit the cluster ( )
Core : upgrade to current Lucene 5 . 0 . 0 snapshot
Remove animal sniffer from low - level REST client ( )
Merge pull request from colings86 / enhancement / enableRequestCacheDefault
[ ML ] Check histogram interval is a divisor of bucketspan ( elastic / x - pack - elasticsearch )
make painless the default scripting language for ScriptProcessor ( )
Turn DestructiveOperations . java into a Guice module .
Close API might cause index data to be wiped , closes .
Renamed old setIgnoreIndices to setIndicesOptions .
fix CategoryContextMappingTests compilation bugs
Fix field mapping updates with similarity ( )
only log that we delete unused shard if it exists
Multi Match : Wrongly defaults to dis_max instead of bool
modified code according to nik9000 ' s comments .
[ DOCS ] Copy relabel script from elasticsearch
Disable BWC Tests for ComponentTemplateMetadata backport ( )
Checkstyle : Catch wide snippets ( )
Fix InternalAutoDateHistogram reproducible failure ( )
Merge pull request from keety / terms_query_disable_coord
Simplify ScriptModule and script registration ( )
lucene 4 : make sure to apply doc boost only once per field name
HLRC : add support for the wait_for_snapshot ILM action ( )
Add CcrRestoreSourceService to track sessions ( )
Fix throttled reindex_from_remote ( )
[ ML ] test mute for testUpgradeJobSnapshot ( )
Merge pull request from rmuir / java . time
Reformats mget API ( )
refactor sourceIndex out of RollupV2Config ( )
[ ML ] adding new inference model size estimate handling from native process ( )
Use extract - pid when killing , too
Fix RareClusterStateIT Cancelling Publication too Early ( )
Build : Increase the number of failed tests shown in test summary
Prevents exception being raised when ordering by an aggregation which wasn ' t collected
Search case insensitivity - complete the version dance now the back port is complete ( )
[ TEST ] check that primaries succeeded
Merge pull request elastic / elasticsearch from rmuir / home_file_cleanup
Use different G1GC options for small heaps ( )
[ Docs ] Correct default scroll_size for update by query ( )
upgrade to mvel 2 . 1 Beta7
Fixing typo and grammar
Docs : Cleanup meta field docs
Mute RestoreBackwardsCompatIT . testRestoreUnsupportedSnapshots .
Followup from elastic / elasticsearchelastic / elasticsearch - Removal of AnalysisService
[ SETUP ] Added the generated JSON skeleton files
Revert " Mark field in ReindexSameIndexTests as final "
Settings : Add infrastructure for elasticsearch keystore
Merge pull request from DynamicScope / patch - 1
Makes freezing QueryShardContext safer by stopping overrides ( )
Introduce licensed plugins ( )
Remove Strings # splitStringToArray
Docs : Fix build after moving SQL QA
Update packaging test script to support mutlijob builds
Merge pull request from nik9000 / remove_ES_CLEAN_BEFORE_TEST
[ ML ] [ Inference ] document new settings ( )
[ ML ] Add daily_model_snapshot_retention_after_days to job config ( )
search as you type fieldmapper ( )
Remove BlobContainer . move ( ) method ( )
move size value / unit to bytes size value / unit , and create a ten based size value / unit
IndexSettings should not be Null in Mapper . BuildContext
Revert " remove duplicate logic for request resolution and routing verification "
Groovy client doesn ' t bind to any variables outside the closure , closes .
Fix half - float range in SupportedTypeTests ( )
Update GCE discovery documentation
SimpleQueryParser should call MappedFieldType . termQuery when appropriate .
[ DOCS ] Added highlighting to the phrase suggester
JVM GC monitor settings
Added new branch for elasticsearch 1 . 2
Add ReplicaAction for Index Lifecycle ( )
Put Mapping : Fix rare case where the response will timeout ( ack = false ) even though it has been applied , closes .
Catch exception when reading corrupted snapshot .
[ TEST ] Ensure file permission for / etc / sudoers . d / elasticsearch_vars
test : unmuted test
backups don ' t become primaries on node failures , primaries are first allocated and then start initializing from gateway
Document the prohibition on freezing data stream write indices ( )
Fix docs build
Inline global checkpoints
Fixes compile errors caused by merge
Search : Reuse Lucene ' s MultiCollector .
[ UTIL ] Changed , that JSON generator sorts API ` parts ` and ` params `
Merge remote - tracking branch ' origin / master ' into json_strict_duplicate_checks
Fixes get snapshot duplicates when asking for _all ( )
fix log message
[ DOCS ] Starting Elasticsearch ( )
Remove Joda usage from ILM ( )
Remove reference to bats test ( )
Merge pull request from jasontedor / load - average - format
Test : add more verbosity when this test fails
Fix AD / vagrant based tests for
Docs : Wrong class name
Tests : fix to specify path and not rely on CWD
Fix disabling ` allow_leading_wildcard ` ( )
[ ML ] Docs and HRLC for datafeed runtime mappings ( )
[ ML ] Rename data frame analytics maximum_number_trees to max_trees ( )
Remember to think about filter caching when upgrading to Lucene 4 . 5 .
Ids with # in them will cause search failures , also , fail when # is used in a type name , closes .
[ DOCS ] Forms role and privilege requirements as bulleted lists in DFA API docs ( )
wait till the index gets created to create the mappings when recovering from the gateway
Repsect indices options on _msearch ( )
Build : Rename test framework project
Migrate some of the Docker tests from old repository ( )
remove RandomStringGenerator , using random strings from randomized testing
Fixes compilation issue in test
Remove unnecessary unchecked cast
Update doc for 1 . 9 . 0
Decentralize plugin security
Percolation results have changed in master
Stabelize more tests
Initial refactoring for phrase suggester
[ TEST ] use true unique transport address in test to prevent address collision
Add missing ' / / CONSOLE ' to code snippet
Fix IndexShardIT # testMaybeFlush ( )
lang - expressions should use DoubleValuesSource , not ValueSource ( )
Merge pull request
Fix compilation error due to empty < p > tag
Update IDE integration to reflect Java 14 requirement ( )
better handling of mean computation of some facets to handle division by 0 , though , won ' t happen on most facets
Merge pull request from rmuir / augmentation
[ INGEST ] Interrupt the current thread if evaluation grok expressions take too long ( )
Remove HighlighterParseElement
[ DOCS ] Fix ML latency definition ( elastic / x - pack - elasticsearch )
Rename template V2 classes to ComposableTemplate ( )
Merge pull request from camilojd / fix - better - exception - query - num - field - regex
Generate and run tests from the docs
SQL : Lowercase es data type ( mapping ) returned from SQL Commands ( )
Remove assertion in testDocStats on deletedDocs counter ( )
by default , index metadata to be stored in smile format and store binary format mapping and alias filter to improve the cost it takes to persist them
Update search . asciidoc ( )
[ TEST ] Fix PercolatorTests # testSimple2
Force default codec in tests where we rely on the latest lucene defaults
Make maximum number of parallel search requests configurable . ( )
Fix CloseWhileRelocatingShardsIT ( )
Tests : Remove broken tests for now
Added tests for info , ping and index
Merge pull request from rjernst / messy_test
Cleanup unnecessary modules and use current version across the board ( elastic / elasticsearch )
[ DOCS ] update versions in java api module . xml
StackOverflowError when calling BulkRequest # add ( )
[ DOCS ] Change ` detector_rules ` to ` rules ` in ML docs
Detach QueryShardContext from IndexShard and remove obsolete threadlocals
[ test ] Fix help test for _cat / shards
Improves doc values format deprecation message ( )
Add released major logic to version utils ( )
Remove MappedFieldType . useTermQueryWithQueryString ( ) and isNumeric ( ) .
Stabilize SLM REST Tests ( )
LICENSE files for security ( elastic / x - pack - elasticsearch )
test : blacklist the ingest default processors rest test
Correct path . conf for integ - test - zip distribution
passwordfix : This removes the password clearing from the authentication service
Remove Migration Upgrade and Assistance APIs ( )
NPE in ShardStats when routing entry is not set yet on IndexShard
Fix typo in percolate - query . asciidoc ( )
[ ENGINE ] Mark current translog as committed if initial recovery is skipped
[ ML ] Create the ML annotations index ( )
Merge pull request from mpereira / patch - 1
[ TEST ] add name parameter to get_alias in update_alias tests
Allow to add existing cache files in searchable snapshots cache service ( )
[ ML ] Fix job ID in C + + logs for normalize and memory estimation ( )
Test : Remove broken yml test feature ( )
Remove createSampleDocument method and use the sync ' ed index method
prepare for next development iteration
Logging : restrict files loaded as logging configuration based on their suffix
Remove PROTOs from TransportAddresses
more docs
Remove docs for proxy mode ( )
Consolidate more parser creation into ESTestCase
Watcher : Reduce script cache churn by checking for mustache tags ( )
Disabling nodes . stats / 30_discovery . yaml rest test
Core : Check if from + size don ' t cause overflow and fail with a better error .
setgid on / etc / elasticearch on package install
SSL / TLS : Properly test for client auth and allow to require
[ TEST ] Added TESTING . asciidoc with some info on testing
move memory monitor benchamrk to single indexing stress test
Handle method rename in core
get the SSL status from the node settings
Ignore timeouts with single - node discovery ( )
Fix total disk bytes returning negative value ( )
[ DOCS ] Added documentation for the keep word token filter
Merge remote - tracking branch ' dakrone / fix - lenient - overriding '
Fix xpack tests with node info ctor changes
[ API ] correct link for flush_synced docs
make optional flush parameter really optional
[ DOCS ] Correct setting type for ` indices . query . bool . max_clause_count ` ( )
Upgrade to Lucene 4 . 10 . 1
move to 1 . 1 . 0 snap
Closing Update service . bat with a safer RAM value
Fixes compile errors from TimeValue serialisation
add script support for terms facets , allowing to control if a certain term will be included in the facets result ( boolean result ) , or muging of the term
test : use the correct number of bwc nodes for old cluster
[ ML ] Add status and increased estimate to memory usage ( )
Provide more context variables in update scripts
Search : By default , don ' t return the version per hit , unless ` version ` is set to ` true ` , closes .
[ DOCS ] Add data streams to bulk , delete , and index API docs ( )
Merge remote - tracking branch ' es / master ' into feature / ingest
Fix AliasMetaData parsing ( )
Add new setting to disable persistent tasks allocations ( )
Fixed rollover docs test for date math indices
fixes to tests
move template query builder tests back to core ( using mockscript )
Allow up to 10 % unbalance in random allocator tests
Tests : Add tests for missing static indexes , and ensure newly added
Strengthen validateClusterFormed check ( )
Add parsing to Significant Terms aggregations ( )
Temp . disable use of new PagedBytesReference since it breaks HTTP replies with 0 - sized Content - Length
Allow follower indices to override leader settings ( )
When considering the size of shadow replica indices , set size to 0
[ Tests ] Remove redudant rest test added in
update documentation with release 2 . 6 . 0
Tests : Added trace logging for more debug information
Prepare for bump to 6 . 0 . 1 on the master branch ( )
[ DOCS ] Format change password API ( elastic / x - pack - elasticsearch )
Implemented AvgAggregatorTests # testDontCacheScripts and removed AvgIT tests ( )
[ ML ] Fix race condition between multiple job deletion and immediate recreation ( elastic / x - pack - elasticsearch )
bump securemock to 1 . 2 for https : / / github . com / elastic / elasticsearch / pull / 15378
Adding a type with _source or _all enabled fails , when these are disabled in index
[ DOCS ] Synchronize location of Breaking Changes ( )
set the node service on the discovery to get the additional service level attributes ( issue )
release 1 . 6 . 0
Upgrade to Netty 3 . 4 . 0 , closes .
Added Dynamic Index Names Support
[ reindex ] Timeout if sub - requests timeout
even if engineUnsafe ( ) is null we still must update the EngineConfig
Fix SQL CLI tests in windows ( elastic / x - pack - elasticsearch )
use index iteration over iterator
[ Security ] Fix concrete security index name ( elastic / x - pack - elasticsearch )
fixed compiler warnining and removed unused imports
[ Docs ] Add note for date patterns used for index search . ( )
Remove sync flush logic in Engine ( )
Adjust bootstrap sequence ( )
handle null doc id set
Serialize Outbound Message on Flush ( )
Marvel : fix wrong SSL settings resolution
Merge pull request from dadoonet / doc / dbq - java - api
Remove unused environment from anomaly detector classes ( )
Test : added awaitFix to AzureTwoStartedNodesTest . two_nodes_should_run_using_private_ip
SQL : Push limit down to GroupByColumn aggs ( elastic / x - pack - elasticsearch )
Update plan for the removal of mapping types . ( )
Improve translog corruption detection ( )
Create SparseFileTracker with already completed / available ranges ( )
use ConfigurationUtils to read string value from config
[ Docs ] Improve tuning for speed advice ( )
Avoid setting connection request timeout ( )
Docs : Added IDs to the highlighters for linking
Add tests for LocalExporter
SQL : Remove log4j and joda from JDBC dependencies ( )
security : extend usage stats
Merge pull request from jasontedor / vagrant - boxes
Split RolloverStep into Wait and Action steps ( )
Adding downsides for the embedded node client
Tests : Ensure binding on localhost host is consistently ipv4 / v6
Proxy box method to use valueOf .
Refactored Painless link nodes into expression nodes to simplify
docs : use ` doc_value_fields ` fields as alternative for nested inner hits _source fetching
Fix build on Fips
[ ML ] Ensure config index mappings are up - to - date before updating configs ( )
[ Docs ] Fix missing new line
[ ML ] Ignore unknown fields when parsing ML cluster state ( elastic / x - pack - elasticsearch )
Logging : Fix test on windows
Implement Comparable in Version ( )
[ ML ] [ Data Frame ] fixing _start ? force = true bug ( )
fixed ShapeFetchService . closes
Search API : Scroll , closes .
Docs : Improved the translog docs
Require Gradle 4 . 3
Updating to throw IllegalArgument exception for null value coordinates . Tests included .
Build : Fix module / integ - test cluster shutdown task ordering
[ TEST ] minor cleanups to AbstractQueryTestCase
[ Transform ] rename data frame transform to transform for hlrc client ( )
Merge remote - tracking branch ' refs / remotes / dakrone / truncate - loglines '
Tests : Silencing failing node attrs REST test
validate query
fix WatcherF compile error
Remove Redundant Cluster State during Snapshot INIT + Master Failover ( )
re - mutes SmokeTestWatcherWithSecurityIT ( )
More checkstyle fixes
Remove unneeded validation in feature set usage
Percolator docs missing a callout
Deprecate permission over aliases ( )
Add a - v flag to output version information , closes
better support for single value to settings builder where the type gets lost
[ DOCS ] Add ML limitation ( elastic / x - pack - elasticsearch )
[ DOCS ] Fix docs integ tests for agg metric field docs ( )
move client - groovy to an updated location
Make the UpdateRolloverLifecycleDateStep retryable ( )
Remove easy uses of setAccessible in tests .
[ TEST ] Use temp directory to fix script loading on windows
Default stopwords list should be ` _none_ ` for all but language - specific analyzers
allow to pass parameters to custom score script
Fix CCR forget follower docs example
fixed interchanged values in field_data stats fixes
upgrade to jackson 1 . 5
Test : assert that the current thread is not already interrupted
Docs : Another bad asciidoc link
Update readme with release dates
Move geometry indexer to FieldMapper ( )
fix smoke test with security reference to renamed ilm
[ DOCS ] Document ` discovery . seed_hosts ` only supports YAML sequences and arrays ( )
[ DOCS ] Fixes table and code block separators in Watcher documentation ( )
Fixed typo in javadoc
Remove dead code / unused class
[ DOCS ] Added more specific versions to the suggesters
[ doc ] fix master build
Mappings : Make DocumentMapper . refreshSource ( ) private .
Test : fix min value being greater than max value in IndexLifecycleManagerIntegTests
test : also wait for paused state
make sure we keep order in mappings when recovering so there won ' t be reparsing of the mappings on nodes ( introduced with faster recovery work done on gateway )
fix checkstyle line lenght issue
Changed the respnose structure of the percentiles aggregation where now all the percentiles are placed under a ` values ` object ( or ` values ` array in case the ` keyed ` flag is set to ` false `
Search Templates : Adds API endpoint to render search templates as a response
Merge pull request from s1monw / make_security_non_optional
Introduce Java version checker
Docs : Clarify sensitive fields watcher encryption ( )
[ DOCS ] Remove unneeded spaces from custom analyzer snippet ( )
Update id - field . asciidoc
Make jobId and from / size mutually exclusive options ( elastic / elasticsearch )
Mark EQL APIs as stable ( )
Query DSL : Terms filter to allow for terms lookup from another document
Highlighting can return excerpt with no highlights
more groovy client work
[ TEST ] Explicit wait for fielddata breaker to be cleared
Add data stream support to CCR ( )
MetaDataIndexAliasesService wrong get type ( )
Don ' t pull the engine for access to the QueryCachingPolicy
EQL : Introduce tie breaker support ( )
add a version to routing table , so it can be reused when not changed after publishing
Added support for random_score function :
More documentation improvements for fielddata loading .
wait before closing the shard if a snapshot is still in progress
Packaging : Update marker used to allow ELASTIC_PASSWORD ( )
Merge pull request from rmuir / lucene_1701068
[ Monitoring UI ] Cleanup Cluster Overview and Node Summary
Removed needless TestRiver used in RiverTests
docs : add pgp key to repositories page
Packaging : change permissions / ownership of config dir
Alerting : rename alerting package to alerts and fix test .
[ TEST ] add comment bout security manager to bwc test description
adapt cluster stats REST tests after merge
Fix Azure Repository with HTTPs Endpoint ( )
Revert " Raised IOException on deleteBlob ( ) "
Startup script exit status should catch daemonized startup failures
Docs : Fixed typo in windows - as - a - service
[ DOCS ] Changed coming [ 1 . 1 . 0 ] to added in snapshot status
[ DOCS ] Reformat common grams token filter ( )
Add shard ID to failed global checkpoint messages
Fixes ExplainLifecycleRequestTests
Simplifies security index state changes and template / mapping updates ( elastic / x - pack - elasticsearch )
[ DOCS ] Update store docs to indicate mmapfs is now the default on 64 - bit Linux
Migrate Python documentation to elasticsearch - py
add smart name filter wrapping for geo filters to automatically filter by type if as well if provided
Update minimum node version for data streams ( )
Remove size 0 options in aggregations
Don ' t limit recoveries in RoutingTableTests
Plugins : Make rest headers registration pull based
Early return if no global checkpoint listeners ( )
CCR recoveries using wrong setting for chunk sizes ( )
no need for weak keys since readers map gets explicitly cleared
Renamed the scripting language Plan A to Painless .
Search optimisation - add canMatch early aborts for queries on " _index " field ( )
add logging with a reason why the cache is cleared ( filter and field )
Fix jarhell marker to be written on info logging , and document how the
Fixed positive infinity bug that can occur in specific scenarios when score mode average is used .
Merge pull request from gmarz / fix / pipeline - aggs - meta
[ ML ] Add more categorization validations ( elastic / x - pack - elasticsearch )
Deprecate / _xpack / security / * in favor of / _security / * ( )
Upgrade to Lucene - 8 . 7 . 0 ( )
[ TESTS ] Fixed cat . allocation tests to work with one node cluster
Tests : add 0 . 20 index and fix test bugs in assertNewReplicasWork ( )
Add a merge policy that prunes ID postings for soft - deleted but retained documents ( )
Pass down parse context instead of just parser in completion context suggester .
Remove ( read | write ) SortBuilder
Add tasks to build Docker build context artifacts ( )
[ Docs ] Add description of simple query string flags ( )
[ ML ] Rename evaluation metric result fields to value ( )
[ ML ] Set ml ' tp queue sizes to a reasonable level .
Disable snapshot / restore tests that fail on windows with AwaitsFix while working on the fix
clarify why jarhell and dependency license check are disabled , add okhttp mockwebserver dependency
Build : remove explicit targetCompatibility from forbiddenapis config
Remove Redundant Cluster State Update Response Classes ( )
Tests : Remove bad apples from schedule engine tests ( elastic / elasticsearch )
[ Discovery ] Add try / catch around repetitive onSuccess calls
Reenable BWC testing after retention lease stats ( )
Merge pull request from jimferenczi / mapping_field_level_boost
[ ML ] Include node name when native controller cannot start process ( )
Use EnumMap in ClusterBlocks ( )
test : echo log file after the index is created to find out why sometimes on CI the indices of that index don ' t get into a started state .
Merge pull request from nik9000 / default_detect_noop
Remove the ` default ` store type . ( )
Core : Use the provided cluster state instead of fetching a new cluster state from cluster service .
Allow optype CREATE for append - only indexing operations ( )
Discovery : If not enough possible masters are found , but there are masters to ping ( ping responses did include master node ) then these nodes should be resolved .
test : make sure to stop nodes after the tribe node integration test
[ Tests ] Remove accidental logger usage
Correct the way to reference params in painless
Network Settings : Allow to explicitly set ipv4 and ipv4 when using _networkInterface_ notation , closes .
Core : let Lucene kick off merges
close parser when parsing update body
unmuted scan parse tests and don ' t be strict when parse the search type . ( Watcher fails when it sees scan is being used )
Upgrade to lucene - 7 . 0 . 0 - snapshot - a128fcb . ( )
Watcher : Fix NPE when search input did not contain search body ( elastic / x - pack - elasticsearch )
Aggregations Refactor : Refactor Scripted Metric Aggregation
[ DOCS ] Edits create data frame analytics job API ( )
[ DOCS ] Remove unneeded filter from common grams analyze ex ( )
Mute AggregatorTestCase . testSupportedFieldTypes
Restore date aggregation performance in UTC case ( )
Build : Cleanup precommit task gradle code
Limit Netty Workers in NativeRealmIntegTestCase ( )
Correct versions limits for snapshot metadata field ( )
Merge pull request from jasontedor / check - not - null - be - gone
add more simple tests
SQL : Move qa directory into sql directory ( )
Mutes DataFeedJobIT test method that uses painless
Unmute already fixed RarTermsAggregatorTests
Exit gradle run task if any node dies ( )
Aggregations : Buckets can now be serialized outside of an Aggregation
Build : add scm info to pom . xml
put mapping to return the parsed source and an indication if it was ack from all the nodes within the timeout
Fix bwc cluster formation in order to run BWC tests against a mixed version cluster ( )
Make ShardRouting and UnassignedInfo immutable ( )
change to trace level logging
streamline these classes a bit
Eagerly initialize Netty 4
Build : Disable fedora for packaging tests ( )
Ignore empty config paths
Transport : read / write support for list of strings
Merge pull request from nik9000 / unraw_2
Convert repository - * from integTest to [ yaml | java ] RestTest or internalClusterTest ( )
[ Docs ] Add clarification to analysis example ( )
move internal stream facet type lookup to work with bytes
Fix inadvertent rename of systemd tests
Add sonatype snapshot repository
Remove broken ` exec ` build target , replace with something better .
Remove generics from ActionRequest
[ build ] Update maven - invoker - plugin to 2 . 0 . 0
Terms Facets : Allow to specify a set of terms to exclude in the request , closes .
Fix BWC for fileBasedRecovery flag removal ( )
Don ' t lookup version for auto generated id and create
Doc : Drop an extra ' a ' in snapshots . asciidoc ( )
[ reindex ] Switch throttle to Float . POSITIVE_INFITINTY / " unlimited "
SQL : Allow whitespaces in escape patterns ( )
SQL : Fix wrong appliance of StackOverflow limit for IN ( )
don ' t log index missing exception when trying to optimize single shard search request
Add docs on JNA temp directory not being noexec ( )
Fix test error in InternalCardinalityTests # testEqualsAndHashcode ( )
Docs : Add migration guide notes for multicast moving to a plugin
[ TEST ] added additional logging to LoggingConfigurationTests
security : only log if we actually loaded the system key
Correct exclusion of test on JDK 11
Iternal : Removed Payload . ActionResponse in favour of an extra constructor to Payload . XContent
Rename migrate
DOC : Added field data circuit breaker settings
. es_temp_file remains after system crash , causing it not to start again
The binary field shouldn ' t be stored by default , because it is already available in the _source .
change index . store . compress . stored_fields setting to index . store . compress . stored
Move watcher history to data stream ( )
fix failed test that were using the wrong form match query
Lift error finding utility to exceptions helpers
[ DOCS ] Replaces docdir attributes in ML APIs ( )
Merge pull request from jpountz / fix / initial_sizing_of_BytesStreamOutput
Add per - socket keepalive options ( )
add CamelCase support to Suggester where missing
create ` es - 1 . 5 ` branch
Monitoring / Watcher : Load version of templates in a static way ( elastic / elasticsearch )
[ Transform ] add debug logging to investigate ( )
Initial commit ( blank repository )
Enable bw tests for and . ( )
[ Docs ] Fix Java Api index administration usage ( )
Don ' t take static memory of test base classes into account when checking for static leftovers .
Upgrade to lucene - 6 . 4 . 0 - snapshot - ec38570 ( )
[ ML ] only log if forecasts are set to failed ( )
[ DOCS ] Make EQL case - sensitive by default ( )
Mute CreateIndexIT . testCreateAndDeleteIndexConcurrently
In test for , we should check the real file name as we have it : - ) .
Fixed typo in getting - started . asciidoc
Test : update missing body tests to run against versions > = 5 . 5 . 0
remove cloud plugin , reimplement an AWS specific cloud plugin with S3 gateway and EC2 discovery
Removed the watcher executor thread pool as it is redundant now with the async indexing .
Watcher : Ensure all json builders use try - with - resources ( elastic / x - pack - elasticsearch )
Testing : Create rule for starting / stopping ApacheDS
Fix test failure .
Merge pull request from obourgain / fix_parsing_multipercolate
InternalTestCluster shouldn ' t auto heal an active disruption when a new one is set
Update to elasticsearch 2 . 0 . 0 . beta2
[ docs ] [ fix ] ` field ` is no longer an option for the script processor ( )
add core module
Remove ` Settings , put ( Map < String , String > ) ` ( )
[ Monitoring ] Update template version to 6 . 0 . 0 beta1 version ( elastic / x - pack - elasticsearch )
Search : Wrap lucene classes at 140 columns ( )
Merge remote - tracking branch ' danielmitterdorfer / bulk - size - limit '
Added BWC indices for 2 . 0 . 2 and 2 . 1 . 1 and versions 2 . 0 . 3 - SNAPSHOT and 2 . 1 . 2 - SNAPSHOT
Handle RejectedExecutionException for node close ( )
[ TEST ] Silence LocalExporterIntegTests . testExport
Added the watch metadata to the template / script model
Document 5 . 0 mapping changes .
Merge pull request from jasontedor / serial - cluster - state - task - executions
SQL : Add remaining matrix aggregations ( elastic / x - pack - elasticsearch )
[ TEST ] Make filestructurefinder . TimeoutCheckerTests more robust
Deprecate Groovy , Python , and Javascript scripts .
Add Create Repository High Level REST API ( )
Fix line - length violations in ABBSAT . java
Disable all annotation processors
[ DOCS ] organize a bit the repository plugins
Ensure CI is run in FIPS 140 approved only mode ( )
Fix missing renamings .
test : removed checking for percolator actions
Revert " Expose external refreshes through the stats API ( ) "
Adjust serialization version for can contain data
Fix find remote when building BWC
[ TEST ] Fix logic in one branch of random mutate test
Implement Iterator # remove for Cache values iter ( )
Remove unnecessary suppression
[ TEST ] refactor DummyQueryBuilder and corresponding parser
Correct CRUDDocumentationIT . java
Geo : Make ShapeBuilders implement Writable
[ DOCS ] Adds missing icons to ILM HLRC APIs ( )
Remove ` sort_order ` and ` sort_mode ` in favor of ` order ` and ` mode `
Merge pull request from tylerjl / docs / upgrade_guide_plugins
rename reference - > origin
Upgrade to lucene - 5 . 2 . 0 - snapshot - 1680200 .
Mute testReopenDuringCloseOnMultipleIndices
Rename Action to ActionType ( )
Parsing a deeply nested json might fail
[ TEST ] Remove deprecated API
Internal : split shard search request into a local and a transport variant
add 0 . 20 . 3
Attachment Processor : setFieldValue only once as a map
Network : Allow to listen on virtual interfaces .
Revert " Mute IndexAuditTrailTests "
Move TokenService to Standard license ( )
Small refactorings to analysis components ( )
fixed highlighted styled built in schema tags to properly increate from hlt1 to hlt10
MINOR : Remove unused ` IndexDynamicSettings ` ( )
Make cluster health classes immutable and have them implement Writeable
Fix recovery throttling to properly handle relocating non - primary shards ( )
AbstractQueryBuilder to no longer extend ToXContentBytes ( )
adapt to changes upstream
Ensure sendBatch not called recursively ( )
Use ordinals for comparison in GlobalOrdinalsStringTermsAggregator . buildAggregation . Closes
Use correct global checkpoint sync interval ( )
[ Docs ] fix curl examples in Nodes Stats docs
Term Vectors / MLT Query : support for different analyzers than default at field
Inline TransportSearchAction # doExecute
Fix potential race during TcpTransport close ( )
Testing : Fix copying of files in shield integration tests
Document completion suggest breaking changes
fix unlikely bug that can prevent Watcher from restarting ( )
Removed all references to site plugins from plugin docs
Muting org . elasticsearch . client . MachineLearningIT . testEstimateMemoryUsage ( )
Make Geo Context Mapping Parsing More Strict ( )
improve setting response / failure on nodes actions
Fix HistoryIntegrationTests timestamp comparsion ( )
XContentExtraction : Fix state bug
Log warn message if leftover shard is detected
control index audit events through an include / excludes mechanism
[ TEST ] fix possible NPE in ClientYamlTestExecutionContext
add missing header
[ Docs ] Clarify ` fuzzy_like_this ` redirect ( )
Add missing ' s ' to tmpdir name ( )
Internal : Simplify CryptoService api
Update ingest jdocs that a null return value will drop the current document . ( )
simplify method signature
make sure to pass the updated settings from the plugins to the modules with transport client
Wire local unreleased bwc versions more efficient for tests ( )
make GeoPoint parsable in lat / lon json format
Reenable BwC Tests after Merging ( )
Add support for expiration after write to Cache
Whitelist randomUUID in Painless ( )
Mute testHistoryIsWrittenWithSuccess
Fix phrase_slop in query_string query ( )
add debug logging for parameters used for ec2 unicast disco
[ DOCS ] Update ` ignore_unavailable ` default for EQL search API ( )
Renamed top level ` filter ` to ` post_filter ` .
Remove client jar
Added the API definition for deleting an index
Watcher : Fix timeout tests
Adapting to changes in master
Fix Race in Concurrent Snapshot Delete and Create ( )
[ DOCS ] Add " index template exists " API docs ( )
Make soft - deletes mandatory in 8 . 0 ( )
Ensure domain_name setting for AD realm is present ( )
Support ignore_keywords flag for word delimiter graph token filter ( )
Add set_priority action to ILM ( )
Expose logs base path
Mock Transport : Allow to simulate network failures
[ Rollup ] Return empty response when aggs are missing ( )
add 0 . 90 . 4
Remove usage of . watches system index in full cluster restart . ( )
Geo Distance Filter Bounding Box Optimization , closes .
Change numeric data types to use SORTED_NUMERIC docvalues type
Fix the clock resolution to millis in ScheduledEventTests ( )
test : change watcher rest tests to wait for yellow
improve logging of failure when merging
Register the TCP_CONNECT_TIMEOUT setting ( )
Exclude JDK Contents / MacOS directory on Darwin ( )
Add data tiers ( hot , warm , cold , frozen ) as custom node roles ( )
Merge pull request from elastic / GlenRSmith - patch - 1
use imutable object for commit id
Docs : Another bad asciidoc link
Fix typo in Elasticsearch help
[ ML ] [ Data Frame ] reduce audit frequency , change log msg , and level ( )
Add a template parameter to override auto_create_index value ( )
Remove CommonTermsQuery and cutoff_frequency param ( )
Removed . settings from . gitignore - @ jpountz intended
[ DOCS ] Adds links to Getting Started with Security ( elastic / x - pack - elasticsearch )
fixed hunspell test to clean up properly
Add actions and listeners for Configuration .
Do not reference values for filtered settings ( )
Test : Fix search query tests to use correct date , mark one test as awaitsfix
[ ML ] Make get_job_stats robust to missing results indices ( elastic / x - pack - elasticsearch )
test : make sure that tests are actually ran
[ Painless ] Add instance bindings ( )
Merge pull request from joshuar / joshuar - docs / clearer - shard - allocation - filtering
add the first server - side unit test
Merge pull request from mikemccand / remove_iw_max_memory_stat
[ TEST ] test also aggregations and fix indendation
has_child query AVG score mode does not always work correctly
Cleanup : Split service transport service for client & server
Update SLM Retention serialization versions ( )
fix minimum_should_match in query_string builder
[ STORE ] Add simple cache for StoreStats
Change BWC version for VerifyRepositoryResponse ( )
Add SSL / TLS settings for watcher email ( )
Revert " Make update a replication action "
Tests : Remove references to tests . slow and tests . integration
Analysis : Custom analyzer to allow to configure position_offset_gap and offset_gap , closes .
fixed compile error caused by change in core
Mappings : Support dots in field names in mapping parsing
when changing the mapping of the _default_ mapping , do not apply the old _default_ mapping to the new one and also do not validate the new version with a merge but parse is as a new type .
Reindex : log more on rare test failure
Cleaner Handling of Store Refcount in BlobStoreRepository ( )
Merge pull request from jpountz / fix / term_vs_terms_query
[ DOCS ] update old status of plugins
prepare branch 1 . x
[ Test ] disable random dynamic templates for audit index
move to lucene 4 . 2 . 1 and ES 0 . 90
Fix testNoMasterActionsMetadataWriteMasterBlock ( )
Add reserved_ml_user and reserved_ml_admin kibana privileges ( )
rename ElasticsearchResponse to Response and ElasticsearchResponseException to ResponseException
Drop node if asymmetrically partitioned from master ( )
[ Test ] Fix QueryStringQueryBuilderTests . testExistsFieldQuery
Merge pull request from rjernst / plugin_default_config
[ TEST ] Don ' t invoke RoutingNodes . assertShardStats ( ) via asserts many times in tests , just once on each reroute call .
Make - try xlint warning disabled by default . ( )
Disallow logger methods with Object parameter
change track_total_hits on async_search . submit to union ( )
add future support to track current merge
Analysis : Add elision token filter , closes .
Make IndexingMemoryController private to IndicesService
[ Test ] Converts LocalExporterTests . testLocalExporterFlush ( ) as REST test ( elastic / x - pack - elasticsearch )
Omit loading IndexMetaData when inspecting shards ( )
Replace LoadingCache usage with a simple ConcurrentHashMap
Use standard CSS separator .
SQL : Use calendar_interval of 1d for HISTOGRAMs with 1 DAY intervals ( )
Allow a cluster state applier to register an observer and wait for a better state ( )
Remove the dependecy on IndexFielddataService from MapperService .
Revert to older exception message
Support external versioning for deletes arriving before initial update , closes .
Correcting 140 character line length
[ Transform ] improve error handling of script errors ( )
update migration doc for removing gateway . format setting
Disable calling mlockall by default , can be enabled by setting bootstrap . mlockall to ` true ` , closes .
Merge pull request elastic / elasticsearch from elastic / master
[ typo ] LICESE_EXPIRATION_ACTION_MATCHER should be LICENSE_ . . .
Mentioned the name of the icu_analyzer
[ DOCS ] Missing sort values now works for all field types
Change " REST Verb " to " HTTP Verb " ( )
Fix strange compilation failure under jdk7 . . .
update the active directory cert in the testnode keystore
randomize routing value when validating query
Enable ResolverStyle . STRICT for java formatters ( )
[ ML ] Switch from max_running_jobs to xpack . ml . max_open_jobs ( elastic / x - pack - elasticsearch )
[ ML ] Persist / restore state for DFA classification ( )
Remove more usages of . watches system index . ( )
minor fixes
Added create_index as a separate index privilege
Fix Deadlock from Thread . suspend in Test ( )
Update opensaml dependency ( )
[ throttler ] changed the Throttler interface
Fix GeoIpProcessorFactoryTests on windows ( )
[ DOCS ] Make ILM documentation data stream aware ( )
Refactor shard limit allocation decider
Introduce a common base response class to all single doc write ops
[ CORE ] Ensure GroupShardsIterator is consistent across requests
SQL : Properly handle indices with no / empty mapping ( )
add MockPageCacheRecycler in test jar
[ Tests ] SimpleRecoveryLocalGatewayTests . testSingleNodeNoFlush could fail if shards were not started
Create ISO8601 joda compatible java time formatter ( )
Enable IndexActionTests and WatcherIndexingListenerTests ( )
Simplify watcher indexing listener . ( )
Permit EQL feature flag in release builds ( )
Merge pull request from adjust / jdk_docs
Return 0 instead of - 1 for unknown / non - exposed ramBytesUsed ( )
update retry - action to be in line with indices requests ( )
no sec manager for now
Update TESTING . asciidoc ( )
Merge pull request from uschindler / painless_debuggerCleanup
nicer logging
Fix AutoFollowCoordinatorTests ( )
Subclass NIOFSDirectory instead of using FileSwitchDirectory ( )
corrected the use of two synonymous words ( )
Remove leftover class , not needed anymore
Mute testTemplateExists ( )
Remove PROTOTYPEs from suggesters
Remove memlock suggestion from systemd service
Mappings : Remove delete mapping API
[ TEST ] adds TRACE logging to org . elasticsearch . snapshots for the
added Czech Stemmer , closes
Handle unmapped fields in _field_caps API ( )
Update field - names - field . asciidoc ( )
[ DOCS ] Expand logging documentation
Updated Analyzing / Fuzzysuggester from lucene trunk
DISCOVERY : Fix RollingUpgradeTests ( )
Rename static DiscoveryNode # masterNode ( Settings ) to isMasterNode
Add tests for _remote / info API ( elastic / x - pack - elasticsearch )
Improve documentation of dynamic mappings . ( )
clean the cluster state when starting ( handles restarts better )
Watcher : Avoid NPE when local address is not resolvable ( elastic / elasticsearch )
Fixes test after replicas removal
[ Transform ] restructure transform yaml tests ( )
FIX : Fixes after merge fail .
doc
Add Strings . substring ( ) that handles short strings .
Fix Windows named pipes with Java security manager ( elastic / elasticsearch )
Change ' standart ' analyzer to use emtpy stopword list by default .
Merge Marvel 2 . 0
Mappings : Remove ability to set path for _id and _routing on 2 . 0 + indexes
[ DOCS ] Edited Docker install & tweaked Docker compose file . ( )
Search Stats : Add current open searches
Add BoolFilterBuilder # hasClauses to be consitent with BoolQueryBuilder
[ DOCS ] Fix redirects and anchors ( ) ( )
[ DOCS ] Correct ` < enrich - policy > ` parm description for comma - sep list ( )
more to ABQ from LBQ is relevant places
[ ML ] Fix custom timestamp override with dot - separated fractional seconds ( )
remove use of request headers / context for pipeline id in favour of instance members added to IndexRequest and BulkRequest
[ DOCS ] Makes the naming convention of the DFA response objects coherent ( )
Improve test stability
Core : remove built - in support for Lucene ' s experimental codecs
Make reindex - from - remote ignore unknown fields
Document all shard allocation filtering attributes ( )
[ DOCS ] Mute failing test snippet
Create weights lazily in filter and filters aggregation ( )
[ TEST ] properly disable merges
[ DOCS ] Fix movavg images and naming
cleanup ILM qa structure ( )
Speed up writeVInt ( )
Increased timeout in ClusterServiceTests . testTimeoutUpdateTask to 100ms
Merge pull request from s1monw / simplify_indices_field_data_cache
Merge remote - tracking branch ' dakrone / no - cluster - name - in - path '
Work around the fact that AssertionError ( String message , Throwable cause ) is a Java 1 . 7 - only API .
[ TEST ] Use non - deprecated _key instead of _term for aggregation ordering
Make sure the remaining delay of unassigned shard is updated with every reroute
[ DOCS ] Adds production monitoring page ( )
Internal : promptly cleanup updateTask timeout handler
Adjust PluginInfo . LICENSED_PLUGINS_SUPPORT value after backport
Decouple MultiValueMode . ( )
[ CORE ] Free pending search contexts if index is closed
Changed info log into debug log
Correct settings prefix for the crypto thread pool ( )
BootstrapSettings final with private constructor
Add xpack / basic scope to SLM documentation ( )
Missing comma between value types ( )
[ DOCS ] Collapses nested objects in data frame analytics APIs ( )
Problem with default configuration file for unicast , closes .
make sure we don ' t go ahead if no index request holds a pipeline id
Fix Watcher binary compatibility between ES 1 . 5 and 1 . 6
TransportClient instead of Client in the code
Skip enterprise license tests in release build ( )
SQL : Lowercase the datatypes in validation error msgs ( )
Make . zip and . tar . gz release artifacts contain same files .
moar changes
Tests : always run Lucene ' s CheckIndex when shards are closed in tests and fail the test if corruption is detected
Aggregations : DateHistogramBuilder accepts String preOffset and postOffset
suppress extrasfs from integ tests , fix bug in random version util
test : make sure the watch doesn ' t get throttled
Fix String . format to use Locale . ROOT in RestRecoveryAction
[ test ] Test that packages don ' t start service
Fix javadoc comment on Elasticsearch # init
Add internal _primary_term doc values field , fix _seq_no indexing
[ TEST ] Make StoreTest extraFS proof
shield : copy settings to tribe nodes
[ Test ] should be abstract
no need for the mvn repository anymore
Merge pull request from Mpdreamz / fix / task - rest - spec
Docs : Add elastics - rb to the list of community clients
Made ` _source ` parsing in ` top_hits ` aggregation consistent with regular ` _source ` parsing in search api .
Mute failing testLimitConcurrentShardRequests
Fix Broken Error Handling in CacheFile # acquire ( )
[ DOCS ] Added fuzzy options to completion suggester
Enabling testing against an external cluster ( )
Mute test for
Scripting : Allow executable expression scripts for aggregations
remove accented character
Mute QueryProfilerIT # testProfileMatchesRegular ( )
Modify skips for get missing alises tests
when parent mapping is set , fix setting routing to required
[ TESTS ] Add get snapshot status test for partial snapshots
Geo : Add GeoJson parser to libs / geo classes ( )
Fix name of template in allowed warning for DS YML test ( )
[ ML ] Enable overall buckets aggregation at a custom bucket span ( elastic / x - pack - elasticsearch )
[ Logstash ] Add new reserved role for managing LS configs ( elastic / x - pack - elasticsearch )
Merge pull request from rmuir / getClassLoader
[ DOCS ] Fix typo in percentile rank aggregation docs ( )
add full rolling restart recvoery test
SQL : Fix deserialisation issue of TimeProcessor ( )
[ DOCS ] Merges duplicate pages for Kerberos realms ( )
Merge pull request from s1monw / fix_repro_line
Remove non - existent variable from Painless context docs ( )
Add OCI annotations and adjust existing annotations ( )
Replaced absolute URLs in docs with attributes
Include ' sourceUrl ' column in dependencies report even when empty ( )
[ ML ] Avoid assertions on empty Optional in DF usage test ( )
upgrade to aws 1 . 0 . 008
Add UTF - 8 encoding to build . gradle in BuildSrc / . ( )
Speed up interval rounding ( )
Added the API definition for ` _search `
Add a dedicated TransportRemoteInfoAction for consistency ( )
Added enum for script sort type
fix index out of bounds error in KV Processor ( )
[ test ] plugins simple RestIT tests don ' t work from IDE
- Added TestCluster # getInstances ( Class ) method .
SQL : correct index name invocation
Remove norelease
[ TEST ] disable Upgrade YAML tests
Adapt BWC layer checks for Exceptions to include v5 . 0 . 2 support
Remove interfaces for notification services
Fix some type checks that were always false ( )
change to Elasticsearch Confidential license
Test fixtures krb5 ( )
[ ML ] [ TEST ] Clean up jobs after SE test
Report terms and version if cluster does not form ( )
[ DOCS ] Combine ` Search your data ` files ( )
Query DSL : Fuzzy Query - add max_expansions , closes .
[ Test ] Mute IndicesStatsTests
fix test
Add " grant_api_key " cluster privilege ( )
Tests : Increase logging for GetWatchTests
Handle deprecation warnings in " old cluster " parts of BWC tests ( )
Merge pull request from jpountz / fix / remove_Lucene_isEmpty
Cleaned up SEach node .
Allow plugins to plug rescore implementations ( )
[ Build ] use the same execution hint file across the pom file
Remove class that is part of commons - codec ( )
Prevent usage of object based RamUsageEstimator in Completion090PostingsFormat
[ DOCS ] Note clause limit in ` index . mapping . total_fields . limit ` docs ( )
Remove Ranking Evaluation API experimental status ( )
[ ML ] Use LinkedHashMap to preserve order in job and datafeed fields ( )
Minor Rearrangements in Snapshot Code ( )
more work on plugin script
geo point new field mapper with geo distance facet based impl
Remove content type auto - detection from search templates
upgrade to joda 2 . 3
ingest : Don ' t allow circular referencing of named patterns in the grok processor .
Add serial collector bootstrap check
[ DOCS ] Update configuring - metricbeat . asciidoc ( )
Merge pull request from xuzha / single_datanode_early_terminate
Simplify module or plugin file checks
Refactored ` Alert ` and ` Alert . Status `
remove requirement for shards / replicas in allocation check steps ( )
Add a note to query_string docs around spaces in field names . ( )
Suppress warning from background sync on relocated primary ( )
Query DSL : indices query to allow to set a ` no_match_query ` , closes .
Build : Fix ability to ignore when no tests are run ( )
Make integ tests antfile a property , so it can be changed
Fix testDelayVariabilityAppliesToFutureTasks ( )
Merge pull request elastic / elasticsearch from rjernst / reorg
[ ML ] [ DOCS ] Fix substitution in ML docs code snipppet ( elastic / x - pack - elasticsearch )
[ ML ] Rolling upgrade test job configurations with empty strings ( elastic / x - pack - elasticsearch )
Align RolloverStep ' s name with other step names ( )
Merge pull request from areek / fix / rollover_rest_api_spec
Enable packaging tests on suse boxes
add trace logging to unicast discovery
Validate if job index already exists after job validation .
Improved test
Cleans up the BlobContainer interface by removing the ( )
Reenable BWC tests after removing cluster state size ( )
Make finalize step of recovery source non - blocking ( )
In MatchQuery , remove a check for fragile search analyzers . ( )
Typo ( )
* fixed the type check for short
Create ` es - 1 . 5 ` branch
SearchLookup to not require MapperService as a constructor argument ( )
Revert shading for the low level rest client ( )
Track multipart / resumable uploads GCS API calls ( )
Remove deprecated _time and _term sort orders ( )
Build : Fix plugin properties generation when version changes
Add a preface title
Add basic tests for sharded JAR
Add reserved dashboards_only_user role ( elastic / x - pack - elasticsearch )
Fixed
[ TEST ] Remove @ AwaitsFix from testMemoryBreaker ( )
Core : also log byte size and doc count when we log slow merges
[ Tests ] Add unit test for InternalAdjecencyMatrix aggregation ( )
Default number of shards is now 1 instead of 5 ( )
painless : Improve string concat tests ; add state machine check after MethodWriter
Expose fielddata " fields " param in standard in indicesStatsRequest
Cleanup SearchRequest & SearchRequestBuilder
Die with dignity on the network layer
Add API for SSL certificate information ( elastic / x - pack - elasticsearch )
Replace failure . get ( ) . addSuppressed with failure . accumulateAndGet ( ) ( )
Fix SearchableSnapshotsIntegTests . testCreateAndRestoreSearchableSnapshot ( )
Introduce Gradle wrapper
Fixed non - valid JSON ( though ES would accept it )
[ ML ] Add a detector_index field to detectors , and use it for updates ( elastic / x - pack - elasticsearch )
Hijack document APIs for . watches index
Deprecate the sparse_vector field type . ( )
doc
[ ML ] Calculate results and snapshot retention using latest bucket timestamps ( )
Remove Repository Stats API ( )
[ DOCS ] Fix indentation
[ TEST ] fixed TestCluster size ( ) javadocs
[ ci ] Print git revision of elasticsearch to ci log ( elastic / elasticsearch )
Don ' t delete local shard data when its allocated on a node that doesn ' t exists
release 1 . 5 . 0
reduce version bounds for spatial usage info to 7 . 11 ( )
Remove unnecessary indices refresh from cluster . state / 20_filtering . yaml file
