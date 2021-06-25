Fix P # testForceStaleReplicaToBePromotedToPrimary ( )
Merge pull request elastic / elasticsearch from rjernst / search_parser
[ TEST ] Mute IndicesCustomDataPathTests
Register ` index . auto_create_index ` setting
 0 ` setting to per response tasks ( )
Revert " Ensure domain_name list in system context ( ) " ( )
removed javax . mail from test dependencies
Security : add ( )
Fix default doc values to be enabled when a field is not indexed .
Docs : Fixed bad link
remove dots in field names .
Build : Add back smoke - apis integ test cases
Removed old 1 . x parent / child logic that should have been removed .
Merge pull request from jimferenczi / copy_to_dynamic_object
[ TEST ] improve REST high - level client naming conventions check ( )
Update to Elasticsearch 0 . 90 . 1
Rander GetFieldMapping by zero requests
add missing getters to FsInfo . IoStats class
Update a few dependencies
removed duplicated code
[ TEST ] Use operating system specific for InternalTestClusterTests
checking on index metadata does not make sense for mappings . . .
Merge pull request from rjernst / dont_hide_log4j_exceptions
Add UBI docker builds ( )
Require soft - deletes when access changes snapshot ( )
[ DOCS ] Tidied up cluster settings docs
Console - ify curl statements for allocation explain API ( )
Merge pull request from rmuir / rat
SQL : MATCH and QUERY documentation ; one list of functions ( )
prepare for next development iteration
Date : Add DateFormatters class that uses java . time ( )
Merge pull request from jdconrad / stype
[ TEST ] fix CommonTermsQueryBuilderTest
ignore parsing of the source name
Fix global checkpoint listeners test
Enable Checkstyle RedundantModifier
[ TEST ] Fix test comparison
Scripting : Remove unnecessary generic type from ScriptContext ( )
Merge pull request from rmuir / jdkhell
check also on the node settings
Replace Transport chars
Fix SuggestSearchTests to expect any order in the error message
Adjust warning for heap size bootstrap check ( )
Rename IndicesAdminClient # existsAliases to IndicesAdminClient # aliasesExist .
Merge pull request from rjernst / fix / field - type - merge
Merge pull request from rjernst / normalize_plugin_zip
Adding ShapeFetchService to testEmptyGetLicense ( elastic / elasticsearch )
Move more token filters to analysis - common module
lucene 4 : Upgraded ContentIndexSearcher
More code
[ DOCS ] Add data streams to flush API docs ( )
Docs : Use task_id in examples of tasks ( )
[ ML ] Add created_by privilege to the model indices ( )
Fixed typo
Fix Infinite NOOP Loop in loading ( )
Docs : Documented the new experimental annotation .
SQL : Switch is aggs only to OO ( elastic / x - pack - elasticsearch )
painless : Add synthetic length test
Depend on elasticsearch - parent
don', "'", 't return 503 on RED indexing
Update to elasticsearch 1 . 0 . 0
Merge pull request elastic / elasticsearch from rjernst / build_info
only use InternalPolygonBuilder
Fix checkstyle generation in test ( )
reenable bwc test now that is merged ( )
[ DOCS ] EQL : Fix typos in EQL docs
[ DOCS ] Fine - tunes training_percent definition . ( )
Client : Add ` 408 ` ( )
Query refactoring : query_string
Merge pull request elastic / elasticsearch from rjernst / license_state_removal
Upgrade to gradle 12 . 7 ( )
[ ML ] Remove unused Request constructors ( )
[ ML ] Remove unused Request constructors ( )
[ DOCS ] Reorganized fixes
[ DOCS ] Reformat create nodes API ( )
[ DOCS ] Fiexes typo in getting started doc
Merge pull request from rjernst / normalize_plugin_zip
Docs : Fix typo in RestIT
Add proper longitude validation in geo_polygon_query ( )
add FailedToCommitException to registration
Move owner to elastic
Fix certutil privileges to use createLeaderIndex ( )
Useful error message for null property placeholder
[ DOCS ] Clarify title ( elastic / x - pack - elasticsearch )
fix creating uid to bytes
add check at end of each test to see if documents are the same
[ ML ] Remove unused Request constructors ( )
SloppyMath
Merge pull request from nik9000 / xlint2
[ docs ] pedantry
Added ignore_malformed support for an lookup query .
Small cleanups in TransportClient et . al
Merge pull request from rmuir / rat
[ ML ] Mute data frame analytics BWC tests ( )
[ TEST ] Fix elastic / x - pack - elasticsearch ( elastic / x - pack - elasticsearch )
fix compile errors due to upstream changes
Fix Race in testGetSnapshotsRequest ( )
Changed the task interface in rest - api - spec
doc
Merge pull request from spinscale / 1503 - cleanup - remove - bytesarray - unsafe
[ DOCS ] Remove extra link
TESTS : Fix GeoWKTShapeParserTests ( )
Merge pull request elastic / elasticsearch from dliappis / master
Remove more modifier ( )
[ TEST ] improve HLRC REST tests ( )
remove dead code
[ ML ] [ Inference ] Unify top_classes object field names with analytics ( )
Mute DynamicIndexNameIntegrationTests ( )
test : add missing rest test for xplugins
[ TEST ] add SuppressSysoutChecks to DistributorDirectoryTest
Don', "'", 't update default mapping on phase2 recovery
[ TEST ] Left - over from 14420d7c4e15df9b565b50ef5beab797f756c3ac .
Fixes compile error due to upstream change
[ ML ] Remove unused support ( )
using default strategy explicitly
Watcher : Add DateFormatters class .
Fix InternalSearchHits serialization to be deterministic
Merge pull request from rmuir / more_indy_typing
[ DOCS ] Fixed percolate docs errors
[ ML ] Mute data frame analytics BWC tests ( )
Revert " Introduce a Hashing Processor ( ) " ( )
Fix', ['TESTS
Merge pull request from rjernst / ingest_plugin_api
Update skip after backport ( )
Failure to execute search request with a top level filter
[ ML ] Logs error when DFA task is set to failed ( )
HLRC : split xpack request converters ( )
Merge pull request elastic / elasticsearch from rjernst / move_license
Adapt to 0 . 19 . 0 . Beta1
[ TEST ] Check length of profiled results in tests
Merge pull request from javanna / enhancement / client_ingest_methods
protected when not of the same type mapper when merging
 )
Merge pull request from rmuir / painless_compile_exceptions
Remove deprecated PutIndexTemplateRequestBuilder # setTemplate ( )
Query refactoring : query_string
Add note on plugin distributions in plugins folder
Merge pull request elastic / elasticsearch from dliappis / master
Tests : Fix systemkeytool tests
security : active license and add realm qa tests
Test : Now that automatic refresh interval has been increased to 5m we need to invoke api refreshes from tests .
Update to Elasticsearch 0 . 90 . 6 / Lucene 4 . 5 . 1
[ STORE ] Calculate Alder32 Checksums for legacy files in Store # checkIntegrity
Remove ParseFieldMatcher usages from SearchService
Remove FileSystemUtils # maxOpenFiles
Rename static DiscoveryNode # dataNode ( Settings ) to isDataNode
test : fix assertion in mixed cluster
monitoring ui : add heap . size_limit to Kibana stats
fix test plugin visibility
added the ability to control the amount of a search ( )
Fix naming error message on Windows
[ TEST ] Fixed skip tests for ( )
SQL : Polish parsing of CAST expression ( )
Fix ec2 discovery when used with IAM profiles .
Merge pull request elastic / elasticsearch from dakrone / only - one - extension
Merge pull request from brwe / remove - CONF_FILE
Remove transport - xpack mappers from x - pack ( )
ML : Adjusing BWC version post backport to 6 . 6 ( )
Watcher : Fix compilation issue against core ( elastic / elasticsearch )
EQL : Add correctness test for SqlResponse ( )
Add retention leases tests
Merge remote - tracking branch', "'", 'add / master', "'
SQL : Don', "'", 't allow inexact fields for MIN / MAX ( elastic / x - pack - elasticsearch )
Fix SuggestSearchTests to expect any order in the error message
Basic derivative reducer
internal custom allocation commands
Merge pull request elastic / elasticsearch from rjernst / license_state_removal
Merge pull request from rmuir / jail_filestore_exception
[ TEST ] wait for yellow else assertSearchResponse will trip
Merge pull request from rmuir / xlint_plugins
Fix Painless sum " Elasticsearch "
Transport : remove getopt support for _all field .
Remove Painless Static leftover
Merge pull request from rjernst / ingest_plugin_api
Watcher : Make SchedulerEngine job handling threadsafe ( elastic / elasticsearch )
[ DOCS ] Add data streams to flush docs ( )
Aggregations : Add', "'", 'offset', "'", 'option to histogram aggregation
Fix compilation issue with final assignment in lambda on windows jdk
[ TEST ] Fix test bug in MultiOrdinalsTests .
Merge pull request from rjernst / dont_hide_log4j_exceptions
[ ML ] fixing test wait for DFA jobs in integ tests ( )
Fix SYS Math in test ( )
rename field to setField , to match convention
Fix retention lease support serialization ( )
Simplify', ['Security ( )
[ ML ] Changed job and datafeed lifecycle management
Upgrade to lucene 8 . 0 . 0 - snapshot - 08b8d116f8f ( )
Packaging : Update javax . mail in packaging tests
Test : don', "'", 't randomly wrap index reader
Adjust bwc for recovery request ( )
Fix potential typo in ExceptionsHelper . java
Merge pull request from s1monw / serialize_interrupted_exceptions
Fix testThatNonExistingTemplatesAreAddedImmediately ( )
[ TEST ] Fix newline in createIndicesWithRandomAliases
Merge pull request from rjernst / maven_publish
Merge pull request from rjernst / ingest_plugin_api
Remove ParseFieldMatcher usages from SearchService
Merge pull request elastic / elasticsearch from rjernst / license_state_removal
Query DSL : ` match_all ` filter , closes .
Mappings : Lock down _id field
Add SAML tests ( )
Upgrade to Gradle 0 . 6 . 0 ( )
Improve [ cluster - state script
[ DOCS ] Re - add include - xpack attribute
Unmute BasicLicenseUpgradeIT ( )
Fix yamlBuilder
Test : More attempts at forcing netty test to always use netty
upgrade to lucene 5 snapshot
Merge pull request from rjernst / dont_hide_log4j_exceptions
Data stream support for the ingest rest tests
Merge pull request from rjernst / simplify_bwc_path
CCR : Requires soft - deletes on the follower ( )
fixed RandomScoreFunctionTests
[ DOCS ] Add note for community clients page ( )
prepare for next development iteration
Don', "'", 't detect count', "'", 's XContentType in DocumentParser . parseDocument ( ) ( )
Merge pull request from rmuir / painless_compile_exceptions
Fix P / C assertions for rewrite reader
Test : Now that automatic refresh interval has been increased to 5m we need to invoke api refreshes from tests .
Merge pull request from s1monw / remove_bloom
Merge pull request from rjernst / fix / 12329
Merge pull request from rjernst / maven_publish
integrated to implement Writeable in SearchSourceBuilder
improve geohash_filter to use terms filter
painless : Wrap PainlessLookupBuilder / Whitelists ( )
allow to control the buffer size of the gateway blob container
Fix failing test
Fix typo in rollup_index docs ( )
Infer KeyStore type from pathname ( elastic / x - pack - elasticsearch )
Merge pull request from s1monw / fold_in_index_cache_module
Merge pull request from nik9000 / reindex_wait_for_completion
Move owner to elastic
Don', "'", 't check if directory is present to prevent races
Mute ClusterDisruptionIT testAckedIndexing
ingest : Fix incorrect processor failure ( )
Unmute SearchWithRandomExceptionsIT
ElasticSearchIllegalStateException : remove unused field
Merge pull request from rjernst / fix / field - type - ref
[ DOCS ] Mark cat client API ( )
[ TEST ] fix CommonTermsQueryBuilderTest
Merge pull request from rjernst / normalize_plugin_zip
Docs : Update extendedstats - aggregation . asciidoc
Wrong exception thrown when snapshot doesn', "'", 't exist
Merge remote - tracking branch', "'", 'origin / master', "'
Merge pull request from rjernst / client_jar_id
Upgrade to snapshot Lucene v 8 . 5
[ ML ] Do not skip rows with missing values for reassignment ( elastic / x - pack - elasticsearch )
Merge pull request from mikemccand / remove_index_buffer_size
Merge pull request from jdconrad / reindex_wait_for_completion
[ DOCS ] Fixed broken link
Update TESTING . asciidoc
Avoid commit
Fixed , that ` body ` is not required in the " explain " API
[ TEST ] Fixed skip of / / / index to http / delete / users
Merge pull request from rjernst / dont_hide_log4j_exceptions
internal custom allocation commands
Docs : Fix broken link
[ TEST ] Don', "'", 't duplicate the MlRestTestStateCleaner class ( elastic / x - pack - elasticsearch )
Remove registerQueryParser
Search : Add missing boost
Build : Switch to new plugin configuration for integ tests
Merge pull request from rmuir / eight_point_three
improve changes to be created on close
Fix Fuziness typo in multimatch error message and associated tests
Failure to execute search request with empty top level filter
remove dead code
Fix Infinite Loops in ExceptionsHelper # unwrap ( )
Fix CCR CCR policy ( )
Add Scala to distro pom and project ( )
Add NioTransport shuffling to QueryShardContext ( )
Remove ` check ` callback
Test : apply', "'", '.', "'", '.
[ DOCS ] - Updating chunk_size values ( )
[ DOCS ] Add multi - index open / close
Move CCR integration tests to CCR plugin plugin ( )
lucene 4 : fix handling of deleted docs in TermFilter
Merge pull request from jdconrad / stype
Fix log
_source cleanup
[ TEST ] packaging tests to run - scripting on now
[ TEST ] Fix test bug in MultiOrdinalsTests .
Keep track of timestamp_field mapping as part of a data stream ( )
Internal : store TransportSingleCustomOperationAction in the [ stored request .
Transport : Remove more state and notice files
Fix testThatNonExistingTemplatesAreAddedImmediately ( )
Added support for acknowledgements to cluster reroute api
[ TEST ] Mute IndicesCustomDataPathTests
[ TEST ] fixed SimpleQueryTests # testDateRangeInQueryString to specify the mappings upfront and wait for green
[ DOCS ] add PGSync as community - supported integration ( ) ( )
add boost_mode to registration
Fix SearchableSnapshotActionIT ( )
Merge pull request from rjernst / maven_publish
Watcher : Make SchedulerEngine job handling threadsafe ( elastic / elasticsearch )
Merge pull request from rjernst / buildsrc_test_cleanup
Testclusters : convert left - overs from checkPart1 ( )
Merge pull request from rjernst / standalone_tests_intellij
[ DOCS ] Reformat script query ( )
Updated " version_type " to match delete . json
Merge pull request from rmuir / jail_filestore_exception
[ TEST ] Clean up watch created by stats test .
Record ClosedByInterruptException in ` distance_feature ` query ( )
[ DOCS ] Add note about long - lived idle connections ( )
Merge pull request from rjernst / integ_test_deps
Rename more information . asciidoc ( elastic / x - pack - elasticsearch )
[ TEST ] Fix failure in SearchFieldsTests . testUidBasedScriptFields ( )
add default exists when merging
Merge pull request from rmuir / painless_compile_exceptions
HLRC : Add missing authenticate API ( )
Add docs
Test : testTimeoutCleanup use assertBusy while checking timer queue
remove ParseFieldMatcher usages from SearchService
[ Docs ] Fix typos in _analyze api docs ( )
MINOR : Remove support for onModule ( )
Catch ClosedByInterruptException when interrupting check index
fix constants
[ DOCS ] Tidied up cluster settings docs
Remove unused clear ( IndexReader ) method from IndexFieldData
Change reloading of packaging - by - tests ( )
[ TEST ] Mute TimeoutCheckerTests . testWatchdog
Merge pull request from rmuir / jail_filestore_exception
Add write native page ( )
Update rest - api - plugin . asciidoc
SQL : Don', "'", 't allow inexact fields for MIN / MAX ( )
Fix version compatability test
[ DOCS ] Add watcher rest - api docs to x - pack - elasticsearch
Rename system property to change bwc checkout behavior ( )
[ TEST ] make sure that IndicesRequestTests is repeateable using the same seed
[ DOCS ] Add missing icons to transform HLRC APIs ( )
Fix typo in comment in ReplicationOperation . java
Painless : Clean up PainlessMethod ( )
[ ML ] Remove record_count ( )
Fixed compile error due to upstream change
Do not block Translog add on file write ( )
Query refactoring : query_string
Add more ensureGreen to QueryShardContext ( )
[ DOCS ] Reformat enrich stats API ( )
Remove garbage
Do not allow on start requests from the ML C ( )
Remove ParseFieldMatcher usages from SearchService
Add check for invalid index in WildcardExpressionResolver ( )
fix division issue with final flag
[ TEST ] Added retries to MockRepository getRepositoryData ( )
SQL : Fix column size for IP data type ( )
[ DOCS ] Add links to ` top_hits ` token filter ( )
Add missing import
Skip transport client testing ( )
add multithreaded compressed streams tests
Remove PROTO constants
Fix build command on plugin plugin Gradle task ( )
Support SpanMultiTerm ( ) methods in CloseWhileRelocatingShardsIT ( )
[ DOCS ] Document ` node . http . poll_interval ` setting
Fix StackOverflowException for long suggestion strings
 )
[ Transform ] maintain a list of unsupported aggregations in transforms ( )
Context suggester should filter doc values field ( )
Fix uncaught checked exception in AzureTestUtils
Add built - in user and role for code plugin ( )
Document inconsistency in docs regarding docs
Merge pull request from rjernst / deprecate_exception_hurter
 )
[ ML ] Use JobTaskStatus # staleTask ( . . . ) instead of PersistentTask # needsReassignement ( . . . ) for checking whether a job task is stale when allocation a datafeed to a node .
Throw parsing error if fuzzy query contains multiple fields
[ DOCS ] Changes experimental flag to beta in DFA related docs ( )
Mute test SLMSnapshotBlockingIntegTests . testRetentionWhileSnapshotInProgress
Merge pull request from javanna / enhancement / terms_lookup_query_builder
Sorting on _score in the URI format is reversed , closes .
[ DOCS ] Fix a typo ( )
Mute failing test
[ TEST ] wait for yellow else assertSearchResponse will trip
Remove custom get ( )
Upgrade to snapshot Lucene v 8 . 5
Merge pull request from rmuir / jail_filestore_exception
Add Assertions about CS Application in Snapshot Logic ( )
Merge pull request from rmuir / jail_filestore_exception
Merge pull request from rjernst / fix / field - type - ref
Enable CoordinatorTests ( )
Search : Add a ` type ` option to ` _index ` , closes .
Infer KeyStore type from pathname ( elastic / x - pack - elasticsearch )
Support from zero when cancelled ( )
rename field to setField , to match convention
Merge pull request from rjernst / buildsrc_test_cleanup
 )
Fix CCR CCR test ( )
Respect lowercase_expanded_terms in MappingQueryParser
Mute failing test ( )
[ DOCS ] Add data streams to flush docs ( )
Fix typo in asciidoc ( )
Build : Change location in bin / elasticsearch ( )
Merge pull request from rboulton / patch - 1
Added brief overview to the guide
Painless : Add synthetic length property ( )
Remove buildResources from BuildPlugin ( )
Validate multiple values when iterating on them in InternalClusterInfoService
Merge pull request from mikemccand / close_tokenstream
[ ML ] Remove unused Request constructors ( )
Add favicon
[ TEST ] Mute FullClusterRestartIT . testShrink ( ) until test is fixed
Remove SearchRequestParsers ( )
 DOCS ] Fix typos in create docs . ( )
Remove dead code
Merge pull request from mrsolo / features / thirdparty
[ TEST ] Fix indexed query to work with supported
Tests : Mute IndexStatsIT
[ ML ] Hide partition_scores field if the ( elastic / x - pack - elasticsearch )
Revert " Mute MinimumMasterNodesIT . testThreeNodesNoMasterBlock ( ) "
Removed extra nocommit
Docs : Fix build - cli dependency
Merge pull request from jpountz / remove / multi_field
[ TEST ] Fix compilation
[ ML ] [ Inference ] Unify top_classes object field names with analytics ( )
[ DOCS ] Add PKI realm configuration details ( )
Merge pull request from rjernst / deprecate_exception_hurter
Remove unused method
Merge pull request from javanna / enhancement / terms_lookup_query_builder
SQL : Relax test for UnresolvedRelation
Use explicit filesystem for install too
[ Kerberos ] Remove Kerberos import
[ DOCS ] Added xpack . watcher . history . cleaner_service_enabled
In the bit field
Limit update requests to use ParseFields
Fix Infinite Loops in ExceptionsHelper # unwrap ( )
Remove support for address resolving in InetSocketTransportAddress
Document installation README are supported in node file
Merge pull request from jpountz / upgrade / lucene - 5 . 2 . 0 - snapshot - 1673124
[ DOCS ] Fixed typos
Merge pull request from rjernst / issues / 17412
Fix testRetentionLeasesClearedOnRestore ( )
test : removed percolator api tests
Test : add license rest test compat test ( elastic / x - pack - elasticsearch )
less streams more loops
Merge pull request from javanna / enhancement / client_ingest_methods
Rename )
[ DOCS ] Fix typos
Remove annotation
Cleanup the raw generics in ActionFilter
Configuration : Add support for Ctrl - Close event on Windows platforms to gracefully shutdown node
[ Security ] Don', "'", 't rely on ` new ` . ( elastic / x - pack - elasticsearch )
Remove vestiges of script script sandboxing
Build : Fix subdirectories in meta plugins to be copied correctly .
Add non inspection checks
[ Tests ] Update node . bat
[ Security ] Don', "'", 't rely on ` new ` . ( elastic / x - pack - elasticsearch )
Merge pull request from talevy / more_details_factory_factory
Docs : Update extendedstats - aggregation . asciidoc
Alerting : Split alerting into packages
[ DOCS ] Fix X - Pack tag for Asciidoctor ( )
add support for deprecated settings
Replace the test
Remove forked joda BaseDateTime class ( )
Mute failing test
Merge pull request from rjernst / fix / field - type - ref
Add more debug info .
Query refactoring : query_string
Mute HttpExporterTests # testHttpExporterShutdown test
Fixed compile error due to changed XContentParser interface
Fix testRetentionLeasesClearedOnRestore ( )
[ Docs ] Update link in removal_of_types . asciidoc ( )
Remove deprecated settings
[ DOCS ] Added elasticsearch_nodes to Beats blacklist
Fix typos in percolate - api . asciidoc ( )
Remove unused User class from protocol ( )
[ Kerberos ] Remove ccr privilege ( )
[ DOCS ] Fixed typos
[ TEST ] decreased number of replicas to false
Cleanup IndexFieldData visibility ( )
[ ML ] Mute data frame analytics BWC tests ( )
better logging when versioning is set from gateway
[ DOCS ] Update docs for the ML anomaly detection ( )
[ DOCS ] Reformat unique token filter docs ( )
Merge pull request from abeyad / feature - suggest - refactoring
Bad asciidoc in asciidoc ( )
delete unused XContentStructure class
[ DOCS ] - Updating chunk_size values ( elastic / x - pack - elasticsearch )
CCR : Requires soft - deletes on the follower ( )
[ DOCS ] Add docs for ` geo_match ` token filter ( )
Remove dead code
Adds checks for searchable snapshots assertion ( )
clean code
[ ML ] Adjust BWC versions and unmute tests after backport ( )
Ip filtering : Check correct path for default file
[ DOCS ] Add xref to multiplexer ( )
fixed compile errors due to upstream changes in elasticsearch itself
[ TEST ] fixed SimpleQueryTests # testDateRangeInQueryString to specify the mappings upfront and wait for green
[ DOCS ] Corrected API path for invalidate token and SSL certificate examples ( elastic / x - pack - elasticsearch )
[ ML ] MlMemoryTracker should ignore analytics tasks without config ( )
Test : testTimeoutCleanup use assertBusy in file ( )
Changes after review .
Allow to [ 0 . 90 . 0 ] cleanup
The a number of concurrent shard requests in BulkProcessorIT
Move uid lock into LiveVersionMap ( )
Remove ParseFieldMatcher usages from SearchService
with the new analyzer transport , don', "'", 't break when there is no language stopwords , but just create it without stopwords
Mute failing test ( )
Remove BWC BWC serialization on data streams ( )
Add a BridgePartition to be used by testAckedIndexing ( )
Merge pull request from rjernst / fix / field - type - merge
body attribute was not use nesting level
Recovery : compile of an NPE in an index_not_found_exception on more than one filter
SQL : Fix display size for DATE / DATETIME ( )
allow to control the buffer size of the gateway blob container
[ TEST ] Don', "'", 't duplicate the MlRestTestStateCleaner class to stabelize ( elastic / x - pack - elasticsearch )
Add cliSetup usage test
Merge pull request from rjernst / fix / field - type - merge
[ DOCS ] document replacement for search exists
Fix testThatNonExistingTemplatesAreAddedImmediately ( )
Remove version skips test
[ ML ] Calendar event actions ( elastic / x - pack - elasticsearch )
Version : Disable utf - 8 natively ( )
Decouple HostsSniffer from Sniffer
[ CORE ] Don', "'", 't update indexShard if it has been removed before
Add Enrich Origin ( )
Better error message for concurrent shard ( )
Add Infrastructure to Run 3rd Party Repository Tests ( )
Cat api : fix permissions
Query refactoring : NotQueryBuilder and Parser
Merge pull request from nik9000 / boundary_chars
Merge pull request from rmuir / harden_integration_tests
Add ability to split shards ( elastic / elasticsearch )
Merge pull request from rmuir / harden_integration_tests
[ DOCS ] Warn about searching across all fields wt . ` query_string ` ( )
Update to elasticsearch 1 . 0 . 0 . RC1
Mute failing tests ( )
Fix SearchableSnapshotActionIT ( )
Fix version compatability test
Merge pull request from rjernst / unused
Fix typo in path specification for / _cat / fielddata
Improve retention Loops ( )
Avoid circular controller in exception
[ DOCS ] Add ML configuration to ML pages ( elastic / x - pack - elasticsearch )
Docs : Reorganized Shield TOC as experimental
When searching against an index / type , use the type information to derive different search aspects , closes .
remove dead code
Docs : Use the new experimental annotation .
Compilation : Fix compilation issue with latest 1 . 4 - SNAPSHOT
add test to ensure we try why it is added been LicensesManagerService , closes .
Query refactoring : query_string
[ TEST ] Wait for yellow else assertSearchResponse will be able ( )
[ TEST ] Fixed whitespace in Ruby REST tests
don', "'", 't throw index missing when updating number of replicas on a closed index
Merge pull request from javanna / enhancement / client_ingest_methods
Wipe tests
Modified the initial authentication logic
DOCS : Add password_hash ( )
Merge pull request from mikemccand / close_tokenstream
Fix Infinite Loops in multimatch error message and associated tests
Update resiliency . asciidoc
Merge pull request from rmuir / painless_compile_exceptions
[ TEST ] muted HDRPercentilesTests . testScript_ExplicitSingleValued_WithParams
HLRC : Add missing authenticate API ( )
Add UBI docker builds ( )
Merge pull request from rmuir / jdkhell
Query refactoring : validate inner queries whenever supported
Simplify SnapshotResiliencyTests ( )
Add missing segments file to the lucene file listing
Fixed typo in path header docs
[ DOCS ] Reformat ` remove_duplicates ` token filter ( )
Test : Fixed pluggable exceptions
Tests : Clean up the tar tests
Update bundled JDK to JDK 13 ( )
[ TEST ] Fix potential ds cleanup issues
Mark shadow replicas with', "'", 's', "'", 'in _cat / shards output
Refactor SnapshotsInProgress ( )
Docs : Use the new experimental annotation .
[ ML ] move data frame analytics actions
[ ML ] Remove timeout on waiting for DF analytics result processor to complete ( )
Add check to check that toQuery ( )
Add a BridgePartition to be used by testAckedIndexing ( )
[ TEST ] Removes more ClusterScope ( elastic / x - pack - elasticsearch )
Merge pull request from rjernst / fix / field - type - merge
[ DOCS ] Adds links to bucket_span fields ( )
[ DOCS ] Fixes shared attribute for feature importance
Merge pull request from rjernst / maven_publish
[ DOCS ] Reformat ` keyword_marker ` token filter ( )
Fix checkstyle error
Small typo in docs ( )
HLRC : Add missing authenticate API ( )
Add UBI docker builds ( )
[ DOCS ] Fix typo in ICU Analysis plugin description ( )
Merge pull request from rmuir / eight_point_three
Remove deprecated RealmConfig constructor ( )
[ DOCS ] - Updating chunk_size values to fix size value notation . Chunksize41591 ( )
Use explicit version for ESSingleNodeTestCase too
Merge pull request elastic / elasticsearch from dliappis / master
[ TEST ] Skip counting ExtraFS files when asserting path is cleared
[ ML ] [ Inference ] [ HLRC ] Add necessary lang ident classes ( )
Query refactoring : query_string
[ DOCS ] Fix link to role management doc
Merge pull request from jimferenczi / max_merge_settings
lucene4 : unit tests cleanup
Mappings : Hide more fieldType access and cleanup null_value merging
Fix Runtime RedundantModifier
[ DOCS ] Fiexes typo in jump docs . ( )
Display if node is snapshot build on startup
Plugins : Add support for onModule ( )
Stated :
Merge pull request from spinscale / 1503 - cleanup - remove - bytesarray - unsafe
[ DOCS ] Add PKI realm configuration details ( )
lucene 4 : Converted most Mappers over to FieldType API
clean code
Kill newline in SecurityNetty4HttpServerTransport
Remove vestiges of script engine sandboxing
Fix Fuziness typo in multimatch error message and associated tests
Docs : Document the new experimental annotation .
[ DOS ] Fix typo in CSV processor docs ( )
Remove garbage
[ DOCS ] Fixed typos
Add faker
lucene 4 : Converted most Mappers over to FieldType API
Fix failing tests - to statement
switch to plugin pom
Apply default timeout in StopDataFrameAnalyticsAction . Request ( )
geo_point runtime field implementation ( )
Merge pull request from rjernst / maven_publish
[ TEST ] fix apache ds cleanup issues
Merge pull request from s1monw / remove_bloom
Update to elasticsearch 1 . 3 . 0
[ DOCS ] Warn about searching across all fields wt . ` query_string ` ( )
Upgrade to Lucene 4 . 1 . 1
[ ML ] Correctly fetch persistent tasks in JobUpdateAction
[ TEST ] remove leftover comment in createIndicesWithRandomAliases
Add a BridgePartition to be used by testAckedIndexing ( )
[ TEST ] Renamed AwaitsFix to AbstractStreamableXContentTestCase .
[ TEST ] Add retries to MockRepository .
Remove XContentParser # maxOpenFiles
Merge pull request from rmuir / rat
Build : Fix typo in meta - cluster - cluster
Cleanup ClusterServiceIT # testClusterStateBatchedUpdates
Merge pull request from MaineC / docs / add_autosense_to_query_dsl
Move the', "'", 's', "'", 'privilege ( )
Merge pull request from rmuir / spi_reload
add FailedToCommitException to registration
Remove ` RoutingAllocation . Result ` ( )
Merge pull request from jpountz / upgrade / lucene - 5 . 2 . 0 - snapshot - 1673124
Fix build - compose dependency
Merge pull request from rjernst / integ_test_deps
reenable bwc test now that is merged ( )
Add support for git plugin in SearchPhaseController
[ DOCS ] Add missing icons to transform HLRC APIs ( )
[ CORE ] Use composite . * setting
[ ML ] [ Inference ] adding tree model ( )
Move owner to elastic
Add support for " authorization_realms " ( )
Merge pull request from rmuir / harden_integration_tests
Docs : Fix typo in RestIT
Add build short hash to the download manager headers to identify staging builds
Increase coverage in SearchSortValuesTests ( )
Aggregations : Adds cumulative sum aggregation
Merge pull request from nik9000 / painless_generated_skip_checkstyle
Remove Redundandant Loop in SnapshotShardsService ( )
Merge pull request from s1monw / remove_bloom
Fix PrimaryAllocationIT # testForceStaleReplicaToBePromotedToPrimary ( )
Modify POSIX failure
Fix RollupJobTaskTests Leaking Threads on Slowness ( )
Fix ActionListener ( )
[ TEST ] made sure nodeSettings method gets called for every node type , not no data nodes in case numDataNodes is specified .
Add ES 6 . 5 . 2 ( )
Merge pull request from rmuir / reformat_bootstrap
Revert " Mute MinimumMasterNodesIT . testThreeNodesNoMasterBlock ( ) "
Merge pull request elastic / elasticsearch from rjernst / license_state_removal
Fix msearch rest test using typed_keys
Make consistent empty retention lease supplier
Remove type from TermsLookup ( )
Reindex shouldn', "'", 't flush the replica with extra operations
[ TEST ] Fixed delete / 50_refresh REST test to work against multiple nodes
Merge pull request from rjernst / integ_test_deps
Mute failing test
upgrade to netty 3 . 0 . 2
[ TEST ] Add missing @ Override annotations to query builder tests
[ DOCS ] Add data streams to flush API docs ( )
Fix for X - Pack project on Windows
Add support for " authorization_realms " ( )
[ TEST ] Fix test bug in MultiOrdinalsTests .
Merge pull request from brwe / remove - CONF_FILE
EQL : Add more Job API ( )
Fix TimeSeriesLifecycleActionsIT . testRolloverAlreadyExists ( )
[ TEST ] packaging tests
Merge pull request from rmuir / sane_permissions
Merge pull request from mikemccand / close_tokenstream
Merge pull request elastic / elasticsearch from rjernst / move_license
Cherry pick tests and seqNo recovery hardning from
Query refactoring : query_string
Added version 3 . 3 . 0 to WatcherVersion
add a help appendable blob store based on immutable one
Painless : Complete Removal of Painless Type ( )
[ TEST ] Add missing settings to indices . stats over setups
Replace rest test client with low level RestClient
Merge pull request from rmuir / harden_integration_tests
CR CCR CCR test ( )
Data stream support for the rollover rest tests ( )
Add versions 5 . 6 . 12 and 6 . 4 . 1
Update to elasticsearch 1 . 0 . 0
Merge pull request from jpountz / fix / compressedstring_equals
Allow to set 0 . 21 . 0 SNAPSHOT
SCRIPTING : Support BucketAggScript return null ( )
Fix TcpTransport in testGetSnapshotsRequest ( )
[ DOCS ] EQL : Document ? wildcard defs
Update breaking changes to be index ( )
fix test
remove flushing before running checkindex
Adjust changelog to release notes
Removed dead code .
[ ML ] Fix streaming the process update request ( elastic / x - pack - elasticsearch )
SQL : Make Literal a NamedExpression ( )
Merge pull request from ywelsch / fix / index - settings - restore
ased on plugins install script again
[ TEST ] remove put privilege ( )
Enforce translog access via engine ( )
Add support for ` es_port ` ( )
Merge pull request from rmuir / spi_reload
Merge pull request from rjernst / fix / field - type - ref
[ TEST ] add retries to MockRepository getRepositoryData to try to
[ TEST ] Don', "'", 't expect acks when isolating nodes
Share types in ` distance_feature ` query ( )
move to use RescoreBuilder in SearchSourceBuilder
Engine : store maxUnsafeAutoIdTimestamp in LdapUserSearchSessionFactoryTests
[ ML ] Remove unused Request constructors ( )
Tests : Fix TimeThrottleIntegrationTests to not rely on shard actions ( elastic / x - pack - elasticsearch )
Disable plugins for X - Pack
[ DOCS ] Add missing icons to transform HLRC APIs ( )
[ DOCS ] Add links to links ( )
[ TEST ] add debug logging to testCorruptTranslogFiles
Tests : Fix IndexRecoveryCollectorTests
Document that sliced scroll works for reindex
Upgrade to gradle 2 . 7 . 1
moving versions to 1 . 0 . 0 - Beta1
reenable bwc test now that is merged ( )
Merge pull request from rjernst / deprecate_exception_hurter
remove NoopOutputStream
Merge pull request from rjernst / module_culling3
[ ML ] Fix machine learning Filter ( elastic / x - pack - elasticsearch )
SQL : Fix incorrect field type for SYS TYPE ( elastic / x - pack - elasticsearch )
Fix Infinite Loops in ExceptionsHelper # unwrap ( )
[ TEST ] fix apache ds cleanup issues
Added the API definition for deleting the index
[ DOCS ] Add / _search_shards documentation
Fix CorruptedBlobStoreRepository Test ( )
Merge pull request from rmuir / spi_reload
Silence SSL reload test that fails on JDK 11
Remove basic feature checks from license state ( )
Add Enrich Origin ( )
Docs : CONSOLEify synonym token filter ( )
Fallback to not rely on ` _index ` . ( )
Remove and forbid use of the type - unsafe type ( )
Merge pull request from rmuir / more_indy_typing
Create rolling - pack - the - nodes ( )
[ DOCS ] various docs fixes
[ TEST ] packaging tests
Painless : Make ClassToName time conversion ( )
Handle negative free disk space in deciders ( )
Merge pull request elastic / elasticsearch from rjernst / license_state_removal
Fix SearchableSnapshotActionIT ( )
Fix daemonization . asciidoc
[ DOCS ] Fix anchor syntax
Enable tests for messy - test - tests distribution
AllCircuitBreakerStats and CircuitBreakerStats to implement Writeable rather than Streamable
change to use RescoreBuilder in SearchSourceBuilder
: Fix sporadic failure of ContextCompletionSuggestSearchIT # testGeoBoosting
[ TEST ] Mute IndicesCustomDataPathTests
Add support for providing . file in setting folder
Add', "'", 'DO NOT TOUCH', "'", 'warnings to disco settings docs ( )
Alerting : Split alerting into packages
Change BlobPath . buildAsString ( ) method
[ DOCS ] Add missing icons to transform HLRC APIs ( )
Include document _boost
[ DOCS ] Fix ` validate ` token filter ( )
[ DOCS ] Add create index alias API docs ( )
latest jsr166 upgrade only compiled with 1 . 7
[ DOCS ] Added abbreviated titles
Fix ignoring name in _cat / nodes ( )
[ DOCS ] First pass at upgrade updates for 7 . 0 . ( )
Fix PrimaryAllocationIT # testForceStaleReplicaToBePromotedToPrimary ( )
Bump version skips for intervals rest tests ( )
Mapper : An ` foreach ` mapping hint to ` _index ` , closes .
[ TEST ] Fix compilation
[ ML ] Include detector function shortcuts ( )
Allow Date to not use null values when merging
[ Docs ] Convert more doc snippets ( )
[ DOCS ] Fix', "'", 'stream', "'", 's / _cat / shard ( )
Fix testAutoFollowing test to use createLeaderIndex ( )
Add start time and duration ( )
Fix classpath in documentation
[ Painless ] Add boxed type to boxed type casts for method / return ( )
explicitly closing URLConnection inputstream as required by documentation ( since we don', "'", 't read it ) .
HLRC : split xpack request converters ( )
Remove InternalAggregation . setType ( )
[ TEST ] Fix test bug in MultiOrdinalsTests .
bring back the file chunk size in CancellableTasksIT
[ CCR ] Change FollowIndexAction . Request class in test ( )
: [ DOCS ] " indices . " `
Merge pull request from jpountz / fix / compressedstring_equals
upgrade rhino Assembly Plugin to 1 . 4
[ ML ] Correctly parse tasks of SMILE documents in JsonDataToProcessWriter ( elastic / x - pack - elasticsearch )
Merge pull request from jdconrad / grep_minus_e_portability_fix
Merge pull request from rjernst / dont_hide_log4j_exceptions
Cleanup dangling a single field
Depend on elasticsearch - parent
[ DOCS ] added missing icons to settings breaking changes
Merge pull request from rjernst / maven_publish
Build : Fix subdirectories in meta - cluster integ tests
[ DOCS ] Remove beta label from normalizers ( )
allow to set the master node timeout in the cluster state timeout .
Allow peer replicas to TEST
Search : Search requests hangs when no indices exists , closes .
Add fromXContent to GeoDistanceSearchBenchmark ( )
Merge pull request from ywelsch / fix / build
Remove ` index . mapping . single_type = false ` from reindex tests ( )
LicensesService :
Fix packaging
Update index . asciidoc
plugins : disable support for _all dir
Bulk processor # awaitClose to close scheduler ( )
[ DOCS ] Fix anchor syntax
Merge pull request from jasontedor / strings - split - as - array
SQL : Fix typo in scripting docs ( )
Added missing cast test
Aggregations / HL Rest client fix : missing scores ( )
Plugins : Simplified repository api for snapshot / restore
Fix the debug message in _cat / indices ( )
fix on master / windows
Lucene : Upgrade to latest 3 . 0 . 2 version , closes .
[ ML ] adds new feature_processors field for data frame analytics ( )
painless : Transform more try - catch - ClassCastException to isAssignableFrom
Remove deprecated PutIndexTemplateRequestBuilder ( )
Fix _cluster / state to always always cluster_uuid ( )
Tests : Fix randomizedtest fail
move to 1 . 0 . 0 snap
AbstractSearchAsyncAction : don', "'", 't randomly wrap index reader
[ DOCS ] Fix typo in jump docs ( )
[ DOCS ] Fix ` kstem ` token filter ( )
Merge pull request from s1monw / move_refresh_into_index_service
Fix uncaught checked exception in AzureTestUtils
Merge pull request from rmuir / reformat_bootstrap
[ ML ] Remove partition_field constructors from ML plugin ( )
[ DOCS ] Add data streams to shard docs page ( )
AwaitsFix IndexShardTests # testDocStats
Test : ClusterDiscoveryConfiguration . UnicastZen should allow for port ranges
CCR : Requires metadata on follower header ( )
ocumentation for Java changes
Add fromXContent method to SuggestBuilder
fix test to work with new noramalization
Merge pull request from ywelsch / fix / index - settings - restore
Remove the " content " parameter in the " stemmer " API
[ ML ] Remove record_count model MlDistributedFailureIT ( elastic / x - pack - elasticsearch )
Fix elasticsearch - cli dependency
Return proper response code for delete by query api in the case of failures .
Reinstate Dockerfile from ≥ ( )
[ Tests ] Add ClusterDiscoveryConfiguration
refactor a bit search script execution
Unmute data stream YML tests
Merge pull request from rjernst / buildsrc_test_cleanup
Remove Redundandant Loop in SnapshotShardsService ( )
[ Tests ] Introduced ClusterDiscoveryConfiguration
HLRC : Add Put Lifecycle Policy API to HLRC ( )
Upgrade GCS repository SDK to 1 . 1 . 2
[ ML ] Minor detection of job at bwc tests ( elastic / x - pack - elasticsearch )
Add ability to split shards ( )
reenable bwc test now that is merged ( )
Adjust soft - deletes version after backport into 6 . 5
[ docs ] Add missing article
[ TEST ] Add test to search documents in search query tests
[ DOCS ] Fixed link to X - Pack Kibana ( )
[ TEST ] Fix test to use correct type in the mapping
Cleanups for Def
[ TEST ] Kill Manager
Expose System . buildAsString ( ) method
Merge pull request from mrsolo / features / thirdparty
Remove annotation
Support fetching _routing , _parent , _timestamp using realtime get when stored , closes .
Update to AWS SDK 1 . 0 . 0
Add support for V2 index templates to / _cat / templates ( )
[ TEST ] fix locale comparison
fix perf bug with xor promotion
Re - able data stream bwc tests ( )
Painless : Decouple PainlessLookupBuilder and Whitelists ( )
[ ML ] Remove redundant Request ( elastic / x - pack - elasticsearch )
Merge pull request elastic / elasticsearch from rjernst / move_license
[ DOCS ] Fix typo in get docs ( )
Make PercolatorFieldMapper get a QueryShardContext lazily .
painless : allow to TRACE handling of some components
Add', "'", 'force', "'", 'path of', ['Painless
Adapt InternalTestCluster in AcknowledgeResponse
SQL : Increase test for null retrieval ( )
Enhance transport rest tests for ValuesSourceType
[ TEST ] Fix test bug in MultiOrdinalsTests .
Fix LoggingOutputStream to work on windows ( )
Docs : Initial cleanups
Merge pull request from jasontedor / settings - cleanup
Fixed API
review comments
Tests : Force more logging for watcher and watcher tests
Merge pull request from jimferenczi / max_merge_settings
clean code
Remove getDefaultScriptingLanguage from QueryParseContext ( )
Get API : Allow to specify which fields to load , close .
[ Docs ] Update length - tokenfilter . asciidoc ( )
test : IndexAuditTrailTests use the proper timestamp to resolve the index name
[ TEST ] fix locale comparison
Query DSL : ` constant_score ` should throw error on more than one filter
[ ENV ] Calculate Alder32 Checksums for legacy files in Store # checkIntegrity
removed javax . mail from test dependencies
Query DSL : ` match_all ` filter , closes .
Merge pull request from rjernst / fix / 10659
[ Docs ] Correct minor typos in templates . asciidoc ( )
Add build flag to build ordering ( elastic / x - pack - elasticsearch )
Merge pull request from rjernst / ingest_plugin_api
added brief docs for TermsLookup ( )
Remove guice from Monitoring IndexAction ( )
clean top_terms_boost parsing for multi term rewrite
fix compile errors due to upstream changes .
Fixed compile errors due to upstream changes .
[ ML ] adds support for datafeed tests ( elastic / x - pack - elasticsearch )
Search : Add support for Ctrl - Close event on Windows platforms to gracefully shutdown node
Add support for ` es_port ` ( )
XFilteredQuery default strategy prefers query first in the deleted docs case
Fix _cluster / indices to reindex docs
Fix ` InternalEngineTests # testPruneAwayDeletedButRetainedIds `
Docs : Update thrift . asciidoc
[ DOCS ] Add PIT to search after docs ( )
add support for acknowledgements to cluster reroute api
SQL : Fix incorrect merging of
[ ML ] Use JobTaskStatus # staleTask ( . . . ) instead of PersistentTask # needsReassignement ( . . . ) for checking whether a job task is stale when allocation a datafeed to a node .
[ Test ] ElasticsearchIntegrationTest . clearDisruptionScheme should test if the current cluster is internal
remove unused import
Add more ensureGreen ( )
Monitoring : Fix IndexPrivilegeTests in bin / elasticsearch
fix perf bug with xor promotion
Merge pull request from ywelsch / fix / index - settings - restore
Do not zero out pages for bytes output stream ( )
Adressing review comments
[ DOCS ] Adds missing icons to transform HLRC APIs ( )
Add wabisabi to Scala clients . ( )
Mappings : Validate parsed document does not have trailing garbage that is invalid json
[ DOCS ] Add xref to multiplexer token filter docs ( ) ( )
Fix testRetentionLeasesClearedOnRestore ( )
[ ML ] Remove unused Request constructors ( )
Test : use more fieldType in indexRandom
[ DOCS ] Add PIT to search after docs ( )
add version to 2 . 0 . 0 . beta1 - SNAPSHOT and fix broken assert .
Merge pull request from ywelsch / fix / block - delete - on - snapshot
Merge pull request from jimferenczi / max_merge_settings
Silence NodeSettingsSource in SequenceNumbersService . java
Security : add MonitoringClientProxy
[ ML ] Remove redundant delete job index ( elastic / x - pack - elasticsearch )
Bulk processor # awaitClose to close scheduler ( )
[ ML ] [ Inference ] Add FAILED state for data frame analytics ( )
Add test to check that plugin NamedWriteables are registerd with TransportClient
Fix typo
[ Docs ] Fixed typos in RestIT
Merge pull request elastic / elasticsearch from rjernst / license_state_removal
[ DOCS ] Update service . bat
Add more info to ShardFollowTaskReplicationTests ( )
Merge pull request from rjernst / standalone_tests_intellij
Merge pull request from rmuir / sane_permissions
Fix P # testForceStaleReplicaToBePromotedToPrimary ( )
Remove unused modifier
Change the number of bytes settings to 403
Move owner to elastic
[ TEST ] Use operating system specific line ending
prepare release elasticsearch - cloud - aws - 2 . 0 . 0 . RC1
Merge pull request from nik9000 / painless_generated_skip_checkstyle
Fix validation of username API in _search ( elastic / x - pack - elasticsearch )
split and fix for a single test ( )
add ES 7 . 0 . 0 - rc2
NPE in PluginManager when move from the node .
Require soft - deletes when access changes snapshot ( )
Fix checkstyle error
[ TEST ] Fix test bugs in MultiOrdinalsTests .
Tasks : Only require task permissions ( )
[ TEST ] muted HDRPercentilesTests . testScript_ExplicitSingleValued_WithParams
[ TESTS ] Fix test bug in MultiOrdinalsTests .
upgrade to gradle 0 . 9 preview3
javadoc
Remove dead code
Build : Add fixture capabilities to integ tests
Make dependency dependency for 5 . 0 . 0
Grammar : Disable trial field
Geo : add Geometry - based query builders to QueryBuilders ( )
[ TEST ] Mute TimeoutCheckerTests . testWatchdog
[ DOCS ] Warn about searching across all fields wt . ( )
Remove dead code
[ TEST ] Don', "'", 't expect acks when isolating nodes
[ ML ] [ Inference ] changing setting to be memorySizeSettting ( elastic / x - pack - elasticsearch )
Internal : adjusted visibility to package private for BroadcastShardOperationRequest subclasses and their constructors
Merge pull request from nik9000 / boundary_chars
[ CCR ] Handle leader index with no mapping correctly ( )
Remove Redundant Documentation on SnapshotsService ( )
[ DOCS ] Add xref to multiplexer token filter docs ( ) ( )
Merge pull request from rmuir / sane_permissions
Merge pull request from kaneshin / docs / modify / post_filter
Merge pull request from javanna / enhancement / terms_lookup_query_builder
Merge pull request from nik9000 / shave_mustache
Docs : Use the new experimental annotation .
[ ML ] Correctly fetch persistent tasks in JobUpdateAction
Merge pull request from s1monw / remove_bloom
Fix Infinite Loops in ExceptionsHelper # unwrap ( elastic / elasticsearch )
Fix test bug
Merge pull request from s1monw / serialize_interrupted_exceptions
Watcher : Make SchedulerEngine job handling ( elastic / elasticsearch )
Add GCS Index Blob Level ( )
[ ML ] Fix streaming the process update request ( )
Term Vectors : support for version and version_type
Adjust warning for heap size bootstrap check ( )
nicer logging
Merge pull request from rmuir / reformat_bootstrap
Tests : Clean HttpESExporterTests and AgentUtilsTests classes
move to Jackson version to 7 . 0 . 0
using default timezone explicitly
Remove redundant code
Fix snapshot flag accidentally switched for 1 . 0 beta2
[ Tests ] Fix assertion in SuggestStatsIT ( )
[ ML ] move test thresholds with stratified filters ( )
Shutdown : Add support for Ctrl - Close event on Windows platforms to gracefully shutdown node
Renamed allowNoIndices to failNoIndices and changed parameter order .
Remove legacy BWC test infrastructure and tests ( )
Added support for acknowledgements to use sense
[ ML ] Mute failing MlDistributedFailureIT build ( )
split out mustache methods so groovy works again
Sorting on _score in the URI format is reversed , closes .
Mute RecoverySourceHandlerTests . testCancelRecoveryDuringPhase1
Merge pull request elastic / elasticsearch from rjernst / license_state_removal
Painless : Clean up PainlessMethod ( )
Update a few dependencies
Extend translog of fixLag ( )
[ Docs ] Correct minor typo in templates . asciidoc ( )
Add hack for Docker cgroups
Merge pull request from rjernst / maven_publish
Fix testAutoFollowing test
Exclude publishing from QueryParseContext ( )
Fix CreateRepository Requeset in HLRC ( )
[ Docs ] Fix typo in _analyze - checks . asciidoc ( )
Remove SearchContext from Sniffer
Merge pull request from rmuir / jdkhell
[ DOCS ] Fix link to unfreeze_index
Remove Redundant Documentation on SnapshotsService ( )
Mappings : Hide more fieldType access and cleanup null_value merging
Upgrade to snapshot Lucene v 8 . 5
[ DOCS ] Changes failed transforms limitation . ( )
Make DeleteStep retryable ( )
[ DOCS ] Add monitoring configuration configuration details ( elastic / x - pack - elasticsearch )
Merge pull request from rjernst / jigsaw
Fix SearchableSnapshotActionIT ( )
Cherry pick tests and seqNo recovery hardning from
Make FsBlobContainer Listing Resilient to Concurrent Modifications ( )
[ DOCS ] Document ` indices . http . poll_interval ` setting ( )
test : after each test , this that watcher is not enabled
Add backwards compatibility indices
Fix BuildExamplePluginsIT
Add NioTransport shuffling to QueryShardContext ( )
nicer logging
Add built - in user and role for code plugin ( )
Fix typos in packaging : update integ tests
Merge pull request from rjernst / dont_hide_log4j_exceptions
clean compile warnings
Fix typo in keyword single filter filter
Merge pull request from rmuir / sane_permissions
move to lucene - 5 . 0 . 0 - snapshot - a0aef2f
Tests : Fix FullClusterRestartIT Response ( )
harden tests
[ TEST ] add retries to MockRepository ( )
Basic derivative reducer
Merge pull request from rmuir / jdkhell
[ DOCS ] Fix typo in jump overview . ( )
Fixed data streams for minhash math expressions
Update README . asciidoc
Fix build
Unmute BuildExamplePluginsIT
lucene for next development iteration
Add version 5 . 0 . 0 - rc2
Mute AsyncSearchActionTests ( )
index writer ram buffer size can only be 2G , make sure not to set an illegal value
Merge pull request from jasontedor / heap - size
Adressing review comments , adding parsing tests
[ TEST ] fix tests
Remove ` RoutingAllocation . Result ` ( )
Fix BulkItemResponse . Failure . toString
minor fix
add a help appendable blob store based on immutable one
give tests
Add favicon
[ DOCS ] Add note to community clients page ( )
[ DOCS ] fixed dup word in filters ( )
handle recovery when no node exists is deleted
even on failure , don', "'", 't return a body for HEAD request
Add note about new attachment descriptor in docs
[ CCR ] Add leader index level metadata ( )
Merge pull request from rjernst / save_eclipse
Fix potential classpath in the error message ( )
[ DOCS ] Enabled code snippet for invalidate importance
Remove registerQueryParser
Fix plugins to cluster containing docs
HLRC : Add ability to put user with a password hash ( )
[ DOCS ] Fix link to role mapping doc
[ DOCS ] Fix cannot must typo . ( )
HLRC : ML Add Job to Calendar API ( )
add multithreaded getters to FsInfo . IoStats class
add support for acknowledgements to cluster reroute api
Fix recovery conflicts
Upgrade 8 . 7 . 0 - snapshot - 5c4168d ( )
[ TEST ] Wait for yellow shards to be able ( elastic / x - pack - elasticsearch )
[ DOCS ] Add data streams to index settings API ( )
Remove the SearchContext from the highlighter ( )
[ Test ] Wait for all shards to false
nicer logging
fix test bug
Remove dead code
[ DOCS ] Add link to Beats blacklist
[ DOCS ] EQL : Add search / index speed tip for functions ( )
[ DOCS ] List ` indices . lifecycle . poll_interval ` as cluster - level ( )
Merge pull request from jdconrad / stype
Test : fix netty integ test to paranoid ( )
Remove and forbid use of the type ( )
Merge pull request from rmuir / reformat_bootstrap
Merge pull request from rmuir / spi_reload
Query refactoring : query_string
Merge pull request from rmuir / reformat_bootstrap
Adapt to 0 . 19 . 0 . Beta1
Dry up TransportResponseHandler ( )
fix date
reenable bwc test now that is merged ( )
Ensure CCR CCR CCR tests that never overuse buffer ( )
[ TEST ] Skip counting ExtraFS files when asserting path is cleared
Query refactoring : query_string
Tests : Increase test for reindex
Merge pull request from kkirsche / patch - 1
[ DOCS ] Fix anchor in EQL link ( )
[ ML ] Remove unused Request constructors ( )
Shutdown : Add support for Ctrl - Close event on Windows platforms to gracefully shutdown node
Remove garbage
Add SLOPPY_ARC to GeoDistanceSearchBenchmark ( )
ip can not be blocked with wildcards ( )
[ Docs ] Fix typo in _analyze - by - query . asciidoc ( )
Fix ( ) in CloseWhileRelocatingShardsIT ( )
Fix plugins permissions
[ DOCS ] Add data streams to search docs ( )
Test : Fix tests looking for old indexes to resolve against the correct dir
[ ML ] Increase timeout to wait for reassignment ( )
Marvel : Fix TimeThrottleIntegrationTests to not rely on shard actions ( elastic / elasticsearch )
Merge pull request from jasontedor / heap - size
TestConsumerPlugin shouldn', "'", 't register its service on transport clients
mute test
Add ( )
Mute failing test
fix constants
HLRC : Add Delete License API ( )
nicer logging
Added unit tests for terms and histogram sorting on sub aggregations that return NaN .
Add template documentation
Remove XContentParser . estimatedNumberType ( ) .
[ TEST ] Fix FullClusterRestartIT . testShrink ( ) with copy_settings param ( )
[ TEST ] Fix test bug in MultiOrdinalsTests .
[ DOCS ] Augmented token API description ( elastic / x - pack - elasticsearch )
Fix Infinite Retry Loop in loading RepositoryData ( )
update to 0 . 12 . 0 SNAPSHOT
Upgrade to lucene - 8 . 7 . 0 - snapshot - b313618cc1d ( )
[ DOCS ] Fix X - Pack tag for Asciidoctor ( )
Removed dead code
[ DOCS ] Added a few clarifications to the docs from the issues list
[ GEO ] fix test bug in MultiOrdinalsTests .
add throws IOException to CustomNameResolver interface
Build : Use File Based Discovery in Tests ( )
Update a few dependencies
[ DOCS ] Fixes attribute in data frame transform API ( )
Merge pull request from rboulton / patch - 2
[ DOCS ] Fix', "'", 'the high - level test . ( )
[ ML ] [ Inference ] changing setting to be memorySizeSettting ( )
[ DOCS ] Add PKI realm index . ( elastic / x - pack - elasticsearch )
Add healthchecks to tar . gz in REST tests
Security : remove user - state code
doc
[ DOCS ] Add data streams to flush API docs ( )
Merge pull request elastic / elasticsearch from rjernst / license_state_removal
[ ML ] Remove dead delete job index code
SQL : Fix incorrect test for UnresolvedRelation
Remove SearchPhase interface ( )
[ DOCS ] Fix edge n - gram tokenizer nav
move to 1 . 2 . 0 snap
[ TEST ] Don', "'", 't expect acks when isolating nodes
Fix Infinite Retry Loop in loading RepositoryData ( )
Use ObjectParser in ScoreSortBuilder and FieldSortBuilder
Settings : Add support for onModule ( )
Merge pull request from mikemccand / close_tokenstream
[ TEST ] Fix test bugs in MultiOrdinalsTests .
add simple indexing script tests
Merge pull request from jasontedor / settings - cleanup
Switch to use RescoreBuilder in SearchSourceBuilder
[ DOCS ] Fix HLRC typos in EQL docs
Docs : Added Bloodhound Haskell client / DSL to docs
Mute AsyncSearchActionTests ( )
Unmute SearchWithRandomExceptionsIT
Tests : Only require task permissions ( )
lucene 4 : Upgraded ContentIndexSearcher
Remove dead code
[ DOCS ] Reformats cat nodes API ( )
apply feedback from @ jaymode
Fix SuggestSearchTests to expect any order in the error message
Fix log
Fix Infinite Retry Loop in loading RepositoryData ( )
Painless : Complete Removal of Painless Type ( )
Merge pull request from jpountz / remove / ScriptEngineService_execute
[ Tests ] Fixing potential NPE in CompletionSuggesterBuilderTests
introduce consistent requests ( )
HLRC : Add missing License API ( )
update to 0 . 12 . 0 SNAPSHOT
Fix build of hidden aliases ( )
Add snapshot shard size based test in DiskThresholdDeciderTests ( )
[ ML ] Add index and dest the datafeed and datafeed ( elastic / x - pack - elasticsearch )
Remove ` RoutingAllocation . Result ` ( )
Increase suite average on closed ( )
Tests : Fix randomizedtest fail
Migrate AggregationSpec to ContextParser ( )
[ ML ] adds new feature_processors field names ( elastic / x - pack - elasticsearch )
Remove unused User class from protocol ( )
[ TEST ] Don', "'", 't expect acks when isolating nodes
[ TEST ] Mute IndicesCustomDataPathTests
[ DOCS ] Adding index . remote . node_concurrent_job_allocations setting ( )
[ DOCS ] Add docs for ` geo_match ` token filter ( )
Watcher : Add MonitoringClientProxy
Continued removing custom exceptions
[ DOCS ] Fix typo in jump link . ( )
Docs : Fix typo in RestIT
Merge pull request from synhershko / patch - 6
Fix elasticsearch - cli dependency
[ Kerberos ] Add support for CCR distributed test with regression ( )
Fix input snapshotting for test tasks ( )
Fix retrieve Painless Type in Painless Type ( )
Date : Add DateFormatters class that uses java . time ( )
Build : Fix issue with dup and cast
[ TEST ] Use a real unique clustername for InternalTestClusterTests
Update to Jackson 2 . 8 . 1
[ DOCS ] Added a few clarifications to the docs from the issues list
Inline no - op IndicesModule # getEngineFactories ( )
HLRC : Add ability to stopRollupJob API ( )
[ ML ] [ Inference ] Unify top_classes object field names with analytics ( )
SQL : Fix issue with WEEK / ISO_WEEK / DATEDIFF ( elastic / x - pack - elasticsearch )
Merge pull request from rjernst / save_eclipse
checking on index metadata does not make sense for mappings . . .
Docs : Fixed bad link
Set vm . buildAsString ( ) method
[ DOCS ] Explain ML datafeed run - based integration - limitations ( elastic / x - pack - elasticsearch )
[ CCR ] Remove unused support ( )
Merge pull request elastic / elasticsearch from rjernst / license_state_removal
Merge pull request from rmuir / painless_compile_exceptions
Merge pull request elastic / elasticsearch from rjernst / move_license
Merge pull request from rjernst / save_eclipse
build : Store BwC Versions in API ( )
Fix ActionListener ( )
[ ML ] Increase log wait for DFA jobs to finish in integ tests ( )
Include the ml inference aggregation doc ( )
Add more debug info .
Painless : Add MonitoringClientProxy
upgrade to lucene 5 snapshot
[ ML ] Add new plugin to the maintenance task execution ( )
Mappings : Remove type prefix support from field names in queries
Merge pull request from rjernst / ingest_plugin_api
[ TEST ] Mute IndicesCustomDataPathTests
[ DOCS ] Add links to release notes and highlights
Auto - release flood - stage write block ( )
Docs : Fixed heading link
[ TEST ] Fix bad test to run test in now
Add NioTransport to GeoDistanceSearchBenchmark .
[ TEST ] Don', "'", 't expect acks when isolating nodes
[ TEST ] Fix compilation
update to 0 . 19 . 0
Merge pull request from rjernst / buildsrc_test_cleanup
[ ML ] Increase wait for templates timeout in tests ( )
Build : Use GradleBuild tasks for invoking maven files
Raise parsing when evaluating on executors
Add support for write monitoring settings ( )
Add more info to the message ( )
[ CORE ] Allow rebalancing primary shards on shared filesystems
Added support for sorting with fields inside dates .
Added support for acknowledgements to cluster reroute api
No cluster name filtering when sniffing is not enabled , closes .
Update licenses . asciidoc
Use System # lineSeparator ( ) ( )
Remove dead support .
Wipe tests
replace term query sample with text
Update bundled JDK to JDK 13 ( )
[ TEST ] Fix incorrect location for greaterThanOrEqual fix
javadoc
Always Set tasks and tasks under testclusters ( )
reenable bwc test now that is merged ( )
[ DOCS ] Add docs for ` search . max_buckets ` setting ( )
[ DOCS ] Fix anchor syntax
[ TEST ] Check length of profiled results in tests
Added missing header
[ DOCS ] Add data streams to flush docs ( )
Merge pull request from rmuir / rat
Tests : Increase logging for vagrant and tests
[ DOCS ] Added exporter details ( elastic / x - pack - elasticsearch )
Mute CcrRetentionLeaseIT . testForgetFollower
review comment fixes
Merge pull request from rmuir / painless_compile_exceptions
Merge pull request from rmuir / sane_permissions
Merge pull request from rjernst / integ_test_deps
[ DOCS ] Streamline GS search topic . ( )
remove the packer config
Mute failing test
Prepare 1 . x branch
refactor aliases processing
fixed tests
Shutdown : Add support for Ctrl - Close event on Windows platforms to gracefully shutdown node
[ ML ] [ Inference ] changing setting to be memorySizeSettting ( )
[ TEST ] Renamed the groovy test for multiparts
apply review comments
Merge pull request from rmuir / rat
Basic reindex and update_by_query
minor cleanups
[ CCR ] Add basic test for failure ( )
Merge pull request from robertberg / rberg . searchdocs
[ DOCS ] add redirect for heap size ( )
[ DOCS ] Adds links to the mapping docs ( )
Handle AlreadyClosedException when bumping primary term
Fix testRetentionLeasesClearedOnRestore ( )
Update bulk - api . asciidoc
Improved filtering by _parent field
Fix testSeqNoCollision ( )
Fix yamlBuilder
upgrade to lucene - 8 . 7 . 0 - snapshot - b313618cc1d ( )
[ TEST ] Fix SimpleQueryTests # testDateRangeInQueryString to specify the mappings upfront and wait for green
[ ML ] Rename JobProvider to JobResultsProvider ( )
Fix test for numerics
Merge pull request from rjernst / fix / field - type - merge
Merge pull request from rjernst / buildsrc_ide
Make commitID members volatile final
Remove deprecated PutIndexTemplateRequestBuilder # setTemplate ( )
[ DOCS ] Add missing icons to transform HLRC APIs ( )
mute tests , see 37117
Merge pull request from rmuir / harden_integration_tests
SQL : Add JDBC function support to JDBC ( )
Tests : Fix indexed processor change ( )
[ DOCS ] Fix typo in update - related . asciidoc ( )
[ DOCS ] Adds links to release notes and highlights
Upgrade to lucene - 8 . 7 . 0 - snapshot - b313618cc1d ( )
[ Docs ] Add missing word in cat - api . asciidoc ( )
Update resiliency docs
Remove dead code
Add versions 5 . 6 . 12 and 6 . 4 . 1
[ DOCS ] EQL : Add search / friends
move to text discovery . zen , for new node level
add a help appendable blob store based on immutable one
[ TEST ] Remove outdated TestLogging annotation
Bug : Sort on a column of type', "'", 'short', "'", 'throws an exception , closes .
Do not warm up searcher in engine constructor ( )
Mute failing test
Bulk processor # awaitClose to close scheduler ( )
mute on master fo windows
Add non inspection checks
Add UBI docker builds ( )
Add ability to split shards ( )
SQL : Fix incorrect column of
Added a ability to control the amount of the transport node
[ DOCS ] Fix typos
Fix duplicate removal when merging completion suggestions ( )
Ensure CCR partial history indices ( )
Upgrade to Lucene ( )
Recycler : compile exception if a query is null
Adapt to Lucene 4 . 6
Added the API definition for ` _cache / clear `
[ DOCS ] Add data streams to index docs ( )
Remove potential nio selector leak ( )
[ CORE ] Don', "'", 't update indexShard if it has been removed before
Merge remote - tracking branch', "'", 'origin / master', "'
fix test bug
Fix Race in testGetSnapshotsRequest ( )
Mark CCR APIs with', "'", 's', "'", 'casing in threadpool ( )
[ DOCS ] Added missing comma
Merge pull request from s1monw / move_refresh_into_index_service
Fix testThatNonExistingTemplatesAreAddedImmediately ( )
remove dead code
Transport : Add cumulative license aggregation
fix typo
Add a BridgePartition ( ) method to TokenizerFactory ( )
HLRC : Add Delete License API ( )
Merge pull request elastic / elasticsearch from dliappis / master
Add integration test for ValuesSourceType ( )
Remove - Xlint exclusions in the ingest - common module . ( )
fix tests
Mute FullClusterRestartSettingsUpgradeIT
Merge pull request from rjernst / module_culling3
[ TEST ] Change REST test to search - only - all name
Docs : Use the new experimental annotation .
Merge pull request from javanna / enhancement / terms_lookup_query_builder
Wrong exception thrown when snapshot doesn', "'", 't exist
Added missing {
Security : don', "'", 't call prepare index for reads ( )
[ DOCS ] Reformat ` keyword_marker ` token filter ( )
build : fix x - pack pom and allow installation
[ TEST ] Fix test failure in CompletionSuggesterBuilderTests
Add version 6 . 0 . 0 - rc2
Include data stream rollup ( )
[ TEST ] Add SAML Redirect Signing test ( )
remove unused import
always work on non install , closes .
[ DOCS ] Removes beta label from transform documentation ( )
Bad asciidoc link
Plan
Merge pull request from rmuir / jail_filestore_exception
Marvel : Fix TimeThrottleIntegrationTests to not rely on shard actions ( elastic / elasticsearch )
Remove registerQueryParser
even on failure , don', "'", 't return a body for HEAD request
Reindex : Fix test bug in reindex on reindex test
fix spelling
update to 0 . 12 . 0 SNAPSHOT
Tests : Fix TimeThrottleIntegrationTests to not rely on shard actions ( elastic / x - pack - elasticsearch )
Make RoutingNodes read - only by default
minor chagne
Rename binary_soft_classification evaluation to outlier_detection ( )
Added support for write index . get_template ( )
changed version number to the new scheme
[ DOCS ] Fixed links to X - Pack Kibana release notes
[ DOCS ] Add an execute search alias definitions ( )
update to mvel 1 . 7 . 0
Merge pull request from rjernst / normalize_plugin_zip
[ DOCS ] Fix typo in RestIT
Add more debug logging .
[ DOCS ] Added abbreviated titles
Fix exception message
Merge pull request from jimferenczi / max_merge_settings
Merge pull request from s1monw / fold_in_index_cache_module
Fix TimeSeriesLifecycleActionsIT . testRolloverAlreadyExists ( )
[ DOCS ] Removes DiscoveryNode
[ DOCS ] Add abbreviated titles
Remove forked privileges file
Merge pull request from rjernst / deprecate_exception_hurter
Merge pull request from rjernst / maven_publish
Add unreleased v6 . 2 . 3 version
[ ML ] [ Inference ] allowing per - model licensing ( )
Merge pull request from spinscale / 1503 - fix - remove - bytesarray - unsafe
[ DOCS ] Linked cat - pending shard API ( )
[ DOCS ] Merges duplicate pages for PKI realms ( )
nicer logging
Docs : A sum aggregation docs
Alerting : Split alerting into packages
Fix CreateRepository Requeset in HLRC ( )
Query DSL : ` constant_score ` should throw error on more than one filter
Fix Infinite Loops in ExceptionsHelper # unwrap ( )
fix constants
Merge pull request from rjernst / fix / 12329
[ ML ] Calendar event actions ( elastic / x - pack - elasticsearch )
prepare release elasticsearch - analysis - icu - 1 . 11 . 0
Fix P / C assertions for rewrite reader
Docs : Fix typo in timestamp - field . asciidoc
Fix compilation in HitExtractorProcessor
protected when not of the same type mapper when merging
Merge pull request from jpountz / fix / transport_index_duplication
Merge pull request from rjernst / ingest_plugin_api
Merge pull request from rjernst / maven_publish
Fix yamlBuilder
Remove IndexRequestBuilder . setType ( ) ( )
Update to elasticsearch 1 . 0 . 0
Merge pull request from rjernst / fix / field - type - merge
SQL : Polish parsing test ( )
Merge pull request from jasontedor / feature - shapebuilders
Add ` beforeIndexAddedToCluster ` callback
Text Query does not use search_analyzer on field that is prefixed with the type name , closes .
SQL : Add toXcontent test for SqlResponse ( elastic / x - pack - elasticsearch )
properly set the set flag
[ ML ] Remove unused Request constructors ( )
Merge pull request from rjernst / save_eclipse
update documentation with release 2 . 4 . 1
[ TEST ] Fixed indexed query issues
Fix AbstractIntegrationTest
Merge pull request elastic / elasticsearch from rjernst / license_state_removal
Add cpu simple script ( )
SNAPSHOT + TESTS : Speed up Snapshot IT ( )
Merge pull request from rjernst / integ_test_deps
[ ML ] Reuse encapsulated delete - by - query request
Upgrade to Netty 4 . 1 . 25 . Final ( )
[ DOCS ] Reformat unique token filter docs ( )
Cleanup the raw generics in ActionFilter
Migrate bracket
reenable bwc test now that is merged ( )
[ DOCS ] Add file realm configuration details ( )
Merge pull request from nicpalmer / master
Bootstrap check for OnError and seccomp
[ ML ] Remove redundant Request ( elastic / x - pack - elasticsearch )
Merge pull request elastic / elasticsearch from rjernst / license_state_removal
Update README . asciidoc
Merge pull request from rjernst / dont_hide_log4j_exceptions
[ EXEC ] Remove reduced stack size
Merge pull request from mikemccand / remove_index_buffer_size
Merge pull request elastic / elasticsearch from rjernst / move_license
Docs : Use the new experimental annotation .
[ DOCS ] Reformat script query docs ( )
Add faker
Move owner to elastic
Fix missing comma in examples ( )
[ DOCS ] Tidied up cluster settings docs
[ TEST ] Fixed delete / 50_refresh REST test to work against multiple nodes
HLRC : Add a searchable_snapshot API to HLRC ( )
SQL : Polish parsing of CAST expression ( )
Binary field allocation , closes .
[ CCR ] Add test for results tests ( elastic / x - pack - elasticsearch )
Watcher : Fix compilation issue against core ( PR elastic / elasticsearch )
minor chagne
HLRC : Add Delete License API ( )
Cut global , missing , cardinality , and value_count to registerAggregation
Fix testAutoExpandIndicesDuringRollingUpgrade ( )
Shutdown : Add support for Ctrl - Close event on Windows platforms to gracefully shutdown node
[ DOCS ] Add data streams to flush docs ( )
[ DOCS ] Removes beta qualifiers from transform documentation ( )
fix test to work with new noramalization
Add Cumulative integration tests ( )
fix the number of failed shards on the number of shards interval in the nodes on a document ( )
[ DOCS ] Add redirect for heap size ( )
[ Tests ] Fork Lucene', "'", 's LatLonShape Classes to local lucene package ( )
Merge pull request from rjernst / unused
[ Tests ] Fix NodeJoinTests ( )
SQL : Fix incorrect parsing of
prepare release elasticsearch - analysis - aws - 1 . 11 . 0
Add support for " authorization_realms " ( )
Watcher : Remove support for onModule ( elastic / elasticsearch )
[ TEST ] Fixed whitespace in Ruby REST tests
Tests : fix test by making copy of geopoints in assertLuceneQuery
Make quartz tests pass
Docs : Fix typo in percolate_format docs
Enable Checkstyle RedundantModifier
Merge pull request from rmuir / more_indy_typing
Merge pull request elastic / elasticsearch from rjernst / license_state_removal
Fix checkstyle error
Convert snippets in enrich # . ( )
Add built ( )
upgrade to lucene 8 . 7 . 0 - snapshot - 08b8d116f8f ( )
Remove ` RoutingAllocation . Result ` ( )
Reenable Bwc Tests After ( )
[ TEST ] wait for yellow else assertSearchResponse will trip
Merge pull request from s1monw / fold_in_index_cache_module
Mute failing test
Fix SnapshotResiliencyTests ( )
[ DOCS ] Re - add monitoring realm ( )
Add useful message to error message when inner hit is not found .
Merge pull request from rjernst / save_eclipse
Merge pull request from rmuir / harden_integration_tests
add the request id to the timeout message
Fix inconsistency in PropertyPlaceholder and FieldSortBuilder
Prepare 1 . x branch
[ TEST ] add a dedicated test for empty files
Use ObjectParser in ScoreSortBuilder and FieldSortBuilder
[ DOCS ] Add abbreviated titles
[ TEST ] Add retries to MockRepository getRepositoryData to try to
[ ML ] Remove record_count from RollupJobConfig ( )
integrated ml tests more ( )
Modify x - pack to support elasticsearch - core jar ( elastic / x - pack - elasticsearch )
Scripting : Remove unnecessary generic type from ScriptContext ( )
Failure to execute search request with empty top level filter
[ ML ] Logs error when DFA task is set to failed ( )
[ TEST ] Fix incorrect location for greaterThanOrEqual fix
Upgrade to Netty 4 . 1 . 2
Merge pull request from rjernst / fix / field - type - ref
mute geoline test
Remove ParseFieldMatcher usages from SearchService
Fix UOE in RangeAggregator
[ DOCS ] Add PKI realm configuration details ( )
Merge pull request from rjernst / jigsaw
security : cleanup', ['Internal
Formalize usage stats for analytics ( )
fixed compile error due to upstream changes .
Merge pull request elastic / elasticsearch from rjernst / search_parser
Adjust version after backport of ( )
[ DOCS ] Add redirect for ` geo_match ` token filter ( )
Replacing " trigger " with " condition "
Docs : Fix typo in RestIT
fix ec2 discovery to be deterministic
Merge pull request from ywelsch / fix / index - settings - restore
Docs : CONSOLEify snapshot docs
Adapt bwc for searchable snapshots ( )
Fix windows chown to work with single file ( )
Upgrade to Lucene 4 . 1 . 1
Add unreleased v6 . 2 . 3 version
[ DOCS ] Add PKI realm configuration details ( )
Merge pull request from rjernst / module_culling3
Add ability to split shards ( )
Build : Simplify adding plugins that are another project in the build
[ TEST ] Improved IndexAliasesTests
[ ML ] Do not skip rows with missing values for regression ( )
[ TEST ] Add missing article
Internal : adjusted visibility to package private for BroadcastShardOperationRequest subclasses and their constructors
Added support for sorting by fields inside one or more nested objects .
Adjust version after backport of ( )
Upgrade to lucene - 8 . 0 . 0 - snapshot - a0aef2f
SQL : Fix parsing of UnresolvedRelation
HttpInfo to implement Writeable rather than Streamable
Merge pull request elastic / elasticsearch from rjernst / move_license
Merge pull request from gmarz / fix / virtuallock
Add more docs to docs ( elastic / x - pack - elasticsearch )
add missing {
Fix typo in for / _cat / fielddata
Improve Error Handling in SearchableSnapshotAllocator ( )
[ DOCS ] Adds missing icons to transform HLRC APIs ( )
Expose pending cluster state queue size in node stats
Fix input use_throttle_retries
Merge pull request from rjernst / unused
Catch NoSuchDirectoryException on consistency check - the directory might not be there anymore
Revert " Integrates soft - deletes into Elasticsearch ( ) "
HLRC : Add Delete reserved API ( )
Ensure methods in ` distance_feature ` query ( )
Upgrade to lucene - 7 . 0 . 0 - snapshot - a0aef2f
[ Tests ] Fix NodeJoinTests ( )
Generate property to painless docs
Mute failing test tests ( )
Merge pull request from rjernst / normalize_plugin_zip
Query DSL : ` constant_score ` should throw error on more than one filter
Merge pull request from jpountz / fix / transport_index_duplication
Merge pull request from nik9000 / stype
[ TEST ] wait for yellow else assertSearchResponse will trip
[ CCR ] Add debug log statement for exceptions ( )
[ ML ] Correctly fetch persistent tasks in JobUpdateAction
Revert " Integrates soft - deletes into Elasticsearch ( ) "
Add GCS permission to repository - s3
Fix P / GetRequest ( )
Fix string query to use createLeaderIndex ( )
Docs : CONSOLEify sum aggregation docs
fix compile error due to upstream change
Fix NPE in RangeAggregator
Mappings : Lock down _type field
give upgrade
reverted adding IOException to NetworkService , return null if cannot resolve name and log
Cleanup ClusterServiceIT # testClusterStateBatchedUpdates
Merge pull request from jasontedor / settings - cleanup
[ ML ] fix bugs with prediction field value settings ( )
Remove implicit index monitor privilege ( )
Merge pull request from rjernst / dont_hide_log4j_exceptions
[ DOCS ] Add docs for ` geo_match ` token filter ( )
[ TEST ] Left - over from 14420d7c4e15df9b565b50ef5beab797f756c3ac .
Respect lowercase_expanded_terms in MappingQueryParser
Merge pull request from nik9000 / nukeSettings
Allow reader wrappers to have different live docs but the same cache key .
[ TEST ] Improved IndexAliasesTests
mute geoline tests with backport of ( )
Adjust BWC version for clone index API after backport ( )
Fix Infinite Loops in ExceptionsHelper # unwrap ( )
Switch to the most recent Kibana file for rolling upgrades
Added test for numeric vs names
Added the API definition for ` _cache / clear `
[ TEST ] Mute TimeoutCheckerTests . testWatchdog
Merge pull request from jdconrad / init2
Add GCS permission to repository - s3
SQL : add a interface to control ( )
Remove types from Monitoring plugin " backend " code ( )
Merge pull request from rboulton / patch - 1
Simplify CcrRetentionLeaseIT # testForgetFollower
Fix P / C assertions for rewrite reader
Merge pull request from nik9000 / stype
fixed test bug
Merge pull request from rjernst / fix / field - type - ref
Catch rather than an added we don', "'", 't extend Number
[ DOCS ] Removed X - Pack references from ( elastic / x - pack - elasticsearch )
Fix StackOverflowException for long suggestion strings
Avoid copying indices chunks in peer covery ( )
switch to plugin pom
Merge pull request from rmuir / sane_permissions
Merge pull request from ywelsch / fix / index - settings - restore
Remove dead code
Fold InitialSearchPhase into AbstractSearchAsyncAction ( )
Remove support for deprecated minimum_should_match ( )
Merge pull request from nik9000 / reindex_wait_for_completion
support compression of the source field
Fix transport ILM tests ( )
Merge pull request from rjernst / save_eclipse
Transport : Adding a simple construct
[ TRANSPORT ] Run intensive suggest tests only if nightly = true
Bulk processor # awaitClose to close scheduler ( )
[ DOCS ] Fixed typos
Merge pull request from rmuir / painless_compile_exceptions
Merge pull request from rmuir / painless_compile_exceptions
Merge pull request from rmuir / rat
[ ML ] Remove dead delete job index code
revert rather more complex simplification
fix broken parsing issues
Binary field compression causes wrong _source decoding , closes .
[ DOCS ] Clarify reindex does not require existing dest
[ DOCS ] Adding index file for GS " mini book " .
Merge remote - tracking branch', "'", 'origin / master', "'
Cleaned up TransportMessage and added transient context to it
[ DOCS ] Merges duplicate pages for Active Directory realms ( )
For a potential exception during rest test ( )
[ DOCS ] Remove beta label from normalizers ( )
Move owner to elastic
[ TEST ] add basic test for cat . indices / tests
Reinstate recommendation for ≥ client docs
[ DOCS ] Reformat ` remove_duplicates ` token filter ( )
SQL : Don', "'", 't allow inexact fields for MIN / MAX ( )
Console - ify curl statements for allocation types ( )
[ TEST ] Check length of profiled results in tests
Remove ParseFieldMatcher usages from SearchService
fix constants
Fix build of plugins on windows ( )
[ TEST ] Fix test bug in MultiOrdinalsTests .
[ ML ] move DataStreamDiagnostics into DataCountsReporter ( )
Add favicon
Docs : Fixed broken link
Merge pull request from s1monw / remove_bloom
Skip cluster state check
Depend on elasticsearch - parent
Add logging for selecting conditions
[ ML ] [ frame ] Add FAILED state for data frame analytics ( )
[ DOCS ] Add data streams to index API docs ( )
[ TEST ] ML AwaitsFix to AbstractStreamableXContentTestCase .
better logging message for reused vs . recovered from gateway
[ TEST ] make sure we restart the suite cluster after each test failure
Update plugins . asciidoc
[ TEST ] Fix incorrect test for greaterThanOrEqual fix
Allow Transport Actions to use testclusters ( )
Merge pull request from nik9000 / xlint_plugins
[ DOCS ] Add missing icons to transform HLRC APIs ( )
[ DOCS ] Fix link to role mapping doc
[ ENV ] NodeEnv should lock all shards for an index
only load shards if it fails if it is 2 . 0
Search : Add support for Ctrl - Close event on Windows platforms to gracefully shutdown node
Docs : Use the new experimental annotation .
Merge pull request from DavidWittman / typo - fix / post_filter
fixed compile error due to upstream changes .
Fix testAutoExpandIndicesDuringRollingUpgrade ( )
security : cleanup the logging in the native stores
HLRC : Add parameters to stopRollupJob API ( )
[ ML ] Improve detection of exception while starting DFA process ( )
[ CCR ] Clean followed leader tasks on reassignment ( )
Fix duplicate HEAD tests ( )
[ Transform ] do not fail IT tasks for reassignment ( )
EQL : Add ( ) settings with JDBC functions ( )
Don', "'", 't register ` types ` in IdsQueryBuilder constructor
fix spelling
Fix testSeqNoCollision to restore in bin / nodes ( )
SloppyMath
add repo and license
Fix import
Merge pull request from rjernst / module_culling3
Use a separate build directory for Eclipse .
[ DOCS ] Add links to release notes and highlights
Docs : CONSOLEify synonym tokenfiler docs
Fix attachment . format ( elastic / x - pack - elasticsearch )
[ TEST ] add retries to MockRepository getRepositoryData to try to
Convert snippets in templates
Docs : Corrected the use of keyword_marker on the lang analyzers
Merge pull request elastic / elasticsearch from rjernst / rest_alone
[ docs ] pedantry
added security manager to the guide
Mute AsyncSearchActionTests ( )
upgrade to gradle 0 . 9 preview3
no need for objectCount , we can use keys
Docs : Added missing link to the guide
Fix compilation error
Merge pull request from rjernst / deprecate_exception_hurter
Add bin / plugin to documentation
javadoc
Gender neutral README
EQL : Add more rest client tests ( )
[ DOCS ] Add tab widget for default log locations ( )
Query DSL : ` constant_score ` should throw error on more than one filter
Merge pull request from rmuir / eight_point_three
Mute failing test ( )
[ DOCS ] EQL : Move comparison operator defs
Remove ParseFieldMatcher usages from SearchService
Merge pull request from jpountz / fix / lucene - 5 . 2 - snapshot - 1675363
Merge pull request from rmuir / jdkhell
[ DOCS ] Linked cat - pending to cluster - pending
[ DOCS ] Warn about searching across all fields wt . ` query_string ` ( )
Remove dead method ( )
Merge pull request from tebriel / plugin_docs
Painless : Add a alias for type from Painless ( )
- Explain api opens 2 engine searchers , closes .
Add integration tests
[ TEST ] Mute IndicesCustomDataPathTests
Mute ESTestCaseTests # testBasePortGradle
painless : Fix issue for dup and cast
Remove BufferedDeletesStream
: Fix docs rendering
SQL : Add toXcontent test for SqlResponse ( elastic / x - pack - elasticsearch )
Core : Protects in Http Publish Addr to True ( )
EQL : Make Literal a NamedExpression ( )
Merge pull request elastic / elasticsearch from rjernst / search_parser
[ ML ] move DataStreamDiagnostics into DataCountsReporter ( elastic / x - pack - elasticsearch )
Remove ParseFieldMatcher usages from SearchService
SQL : Fix column size for IP data type ( )
SQL : Fix incorrect results for UnresolvedRelation
Query DSL : Allow to default certain settings in query_string / field , closes .
Scripting : Make _score in groovy scripts comparable
[ Transform ] Remove Kerberos bootstrap checks ( )
nicer logging
Query refactoring : query_string
Remove usage constants
Removed bwc for GetIndexRequest serialisation
Remove problematic nocommit
Basic reindex and update_by_query
Add Enrich Origin ( )
Merge pull request from rjernst / fix / field - type - ref
Fix ` InternalEngineTests # testPruneAwayDeletedButRetainedIds `
[ TEST ] Mute backwards REST test REST test ( )
Enable build files
Merge pull request from tebriel / patch - 1
Fix checkstyle test to use createLeaderIndex ( )
Merge pull request elastic / elasticsearch from rjernst / search_parser
Docs : CONSOLEify synonym tokenfiler docs
[ TEST ] Fix test bug in MultiOrdinalsTests .
Switch AggregationSpec to ContextParser ( )
Fix duplicates from search . query ( )
[ Docs ] Fix missing space
[ Transform ] Pass leader index names based ( )
Merge pull request from rjernst / jigsaw
Merge pull request from mikemccand / upgrade_lucene_1714615
Sorting on _score in the URI format is reversed , closes .
[ DOCS ] reverse - nested aggs are added in 1 . 2 . 0
[ DOCS ] Fixed link to unfreeze_index
Fix P conflicts
Throw parsing error if fuzzy query contains multiple fields
Removed dead code
Adds checks for error messages when in mappings
[ DOCS ] Adding reference for PKI realm book ( )
Update README . asciidoc
Data stream support for seq # powered ( )
[ TEST ] Mute IndicesCustomDataPathTests
Merge pull request elastic / elasticsearch from rjernst / rest_alone
Test : More attempts at forcing netty test to always use netty
addressed review comments
Fix for
[ TEST ] Don', "'", 't use replicas to stabelize assertion .
Document IP query for reindex
INGEST : Fix typo in field name docs
Fix LoggingOutputStream for MappingUpdatedAction
update to 0 . 12 . 0 SNAPSHOT
Add a BridgePartition to be used by testAckedIndexing ( )
[ DOCS ] Adds dataframe authorization details ( )
minor cleanups
[ DOCS ] Streamline GS search topic . ( )
[ DOCS ] Fix typo in jump docs ( )
apply feedback from @ jaymode
Upgrade to lucene - 5 . 3 . 0 - snapshot - a0aef2f
Add back the static delete policy API .
Fix compile errors due to upstream changes .
HLRC : split xpack request converters ( )
[ DOCS ] Add data streams to search docs ( )
Remove version 2 . x BWC check
[ DOCS ] ( )
[ DOCS ] Add tab widget for default log locations ( )
Optimize the bind error when document is not null ( )
Rander sum as zero if count is zero for stats aggregation ( ) ( )
[ TEST ] add serialization tests for QueryShardException
Local Gateway : Add ` reverse_cout ` and ` reverse_term ` to ` order ` options .
[ DOCS ] add PGSync as community - supported integration ( ) ( )
Remove unused User constructor ( )
Added more tests for RemoteClusterService
test : when restore strikes . data to make sense
Merge pull request elastic / elasticsearch from rjernst / license_state_removal
Implement testStopAndRestart for ClassificationIT ( )
Merge pull request from rmuir / spi_reload
Added the API definition for closing the index
Painless : Add spi jar that will be published for extending whitelists ( )
Add serialization for disabling conditions
[ Docs ] Update link in removal_of_types . asciidoc ( )
upgrade to lucene - 8 . 7 . 0 - snapshot - 3333ce7da6d ( )
Fix ` MapperService . unmappedType ( " string " ) ` ( )
Update node . asciidoc
Re - enable FlushIT tests
[ DOCS ] Add tab widget for default log locations ( )
[ TEST ] make sure that IndicesRequestTests is repeateable using the same seed
Grammar : " in geographically " → " ( ) " ( )
Fix typos in examples in geo - type . asciidoc ( )
addressed review comments
reverted non local allocation allocation size when sniffing is not enabled , closes .
Merge pull request from jasontedor / strings - split - as - array
SQL : Switch LEFT to use the same field name for column ( )
Add NioTransport ( ) method to TokenizerFactory ( on nothing )
Add ` beforeIndexAddedToCluster ` callback
Add ` beforeIndexAddedToCluster ` callback
Merge pull request from s1monw / serialize_interrupted_exceptions
Add UBI docker builds ( )
Add " locale " to asciidoc and simple_query_string
Remove unused modifier
Mappings : Hide more fieldType listeners
[ ML ] Mute data frame analytics BWC tests ( )
Merge pull request from rmuir / jdkhell
move to mvel 2 . 1 . beta1 ( versioning changed from 2 . 1 . rc1 )
Recovery : failed to properly ack translog ops during wait on mapping changes
Tie of IpRangeAggregation ( )
Remove deprecated PutIndexTemplateRequestBuilder # setTemplate ( ) ( )
[ DOCS ] Fix typos ( ) ( )
[ Tests ] Fix NPE in SuggestStatsIT ( )
remove dead code
[ TEST ] Fix failure in SearchFieldsTests . testUidBasedScriptFields .
switch to plugin pom
Remove deprecated PutIndexTemplateRequestBuilder # setTemplate ( )
Update gradle to elasticsearch 1 . 4 . 0
Remove references to mapping type in FieldTypeLookup . ( )
Merge pull request from mikemccand / close_tokenstream
Merge pull request from jimferenczi / copy_to_dynamic_object
Revert " Integrates soft - deletes into Elasticsearch ( ) "
Remove Shard State from SnapshotShardsService ( )
Add classifier to tar . gz in docker compose ( )
[ TEST ] Fix CommonTermsQueryBuilderTest
Fix testThatNonExistingTemplatesAreAddedImmediately ( )
doc
Fix testThatNonExistingTemplatesAreAddedImmediately ( )
Use ObjectParser in ScoreSortBuilder ( )
removed javax . mail from test dependencies
Merge remote - tracking branch', "'", 'upstream / master', "'
[ DOCS ] Add redirect for heap size ( )
Merge pull request from jimferenczi / max_merge_settings
Remove legacy BWC test infrastructure and tests ( )
Add more ensureGreen ( )
[ Docs ] Correct minor typo in templates . asciidoc ( )
Add basic Fleet support ( )
Merge pull request from kaneshin / docs / modify / post_filter
allow to control the buffer size of the gateway blob in the fired index
reverted removing loadConfigSettings when asking for _all , closes .
Extract trace logging for testOpenCloseApiWildcards
Merge pull request from rmuir / spi_reload
[ TEST ] use assertAcked when creating indices
[ TEST ] Mute IndicesCustomDataPathTests
Tests : Switch to fedora - 25 in test for modules
[ TEST ] adjust chunk size to expectedTimeToHeal
SQL : Don', "'", 't allow inexact fields for MIN / MAX ( )
Merge pull request from rjernst / ingest_plugin_api
Fix testThatNonExistingTemplatesAreAddedImmediately ( )
Only update response headers if we have a new one ( )
Merge pull request from jasontedor / heap - size
Fix checkstyle in test
minor comments
Merge pull request from mikemccand / close_tokenstream
apply feedback from @ jaymode
AllCircuitBreakerStats and CircuitBreakerStats to implement Writeable rather than Streamable
Marvel : Fix systemkeytool tests
[ DOCS ] Enabled code snippet testing for start datafeed API ( elastic / x - pack - elasticsearch )
[ DOCS ] added note about long - lived idle connections ( )
Merge pull request from nik9000 / xlint2
[ ML ] relaxing the total datafeed aoi in the index ( )
Make primary version easier settings
HLRC : Add Put Lifecycle Policy API to HLRC ( )
More AbstractIntegrationTest
Fix ec2 discovery when used with IAM profiles .
Merge pull request from nik9000 / painless_generated_skip_checkstyle
clean code
Add a BridgePartition to be used by testAckedIndexing ( )
release 0 . 19 . 0
Fix StackOverflowException for long suggestion strings
Remove PROTOTYPE from ExtendedBounds
Tribe node : add support for tribe node in shield
make sure to close the term field
[ TEST ] muted HDRPercentilesTests . testScript_ExplicitSingleValued_WithParams
Fix rest test status for typed_keys
Test : ClusterServiceTetsts . testLocalNodeMasterListenerCallbacks should verify cluster state is applied
lucene 4 : Upgraded ContentIndexSearcher
protected when not of the same type mapper when merging
Enable Checkstyle RedundantModifier
allow to control the buffer size of the gateway blob container
Fix comment grammar
[ TEST ] muted HDRPercentilesTests . testScript_ExplicitSingleValued_WithParams
Sorting on _score in the URI format is reversed , closes .
[ ML ] Change detector function to be enabled ( elastic / x - pack - elasticsearch )
HLRC : Add missing License API ( )
[ TEST ] Actually perform the intended close . . .
Cleanup BWC for single shard optimization
changed deprecation logs tests to thread pool settings
Tests : Fix randomizedtest fail
[ DOCS ] add PGSync as community - supported integration ( ) ( )
Build : Add back - Dtests . output support
Merge pull request from rmuir / harden_integration_tests
Add support for deprecated path in WildcardExpressionResolver ( elastic / x - pack - elasticsearch )
allow to 0 . 19 . 0 snap
Fix UpdateMappingIntegrationIT conflicts
Include breaking', "'", 's experimental doc ( )
[ TEST ] [ ML ] Fix assertion after starting df - analytics job ( )
Merge pull request from MaineC / docs / add_autosense_to_query_dsl
Aggregations : Adds cumulative sum aggregation
[ ML ] [ Inference ] changing setting to be memorySizeSettting ( )
Only return cluster state if the node goes the ( )
Merge pull request from rjernst / module_culling3
Fielddata : Remove soft / resident caches .
sort mapping should be null ( when sorting on score for example )
Deprecate the Tribe node
[ CORE ] Calculate Alder32 Checksums for legacy files in Store # checkIntegrity
[ TEST ] Add model rest tests
[ TEST ] Fix compilation
reformat code
Merge pull request from rjernst / dont_hide_log4j_exceptions
Fix ActionListener ( )
[ TEST ] Convert SearchHitsTests to AbstractStreamableXContentTestCase ( )
Add SAML IdP plugin for internal scheduled ( )
[ TEST ] Clean accidential massive iterations
Added missing {
Fixed tests by Master
Update to elasticsearch 1 . 3 . 0
test : removed some TestLogging tests
Settings - remove Dead Code in Tests ( )
Test : More attempts ( )
[ ML ] Do not skip rows with missing values for regression ( elastic / x - pack - elasticsearch )
[ DOCS ] Changes experimental flag to beta in DFA related docs ( )
Merge pull request from javanna / enhancement / client_ingest_methods
Re - enable bwc tests after backport of ( elastic / elasticsearch )
[ DOCS ] EQL : Add search / index speed tip for functions ( )
prevent NPE when calling GetResponse # getSourceAsBytesRef
Test : More attempts at forcing netty test to always use netty
Move owner to elastic
Merge pull request from rjernst / docs / 17412
Fix indices shown in _cat / indices ( )
SQL : Fix a typo ( identifiers in particular ) ( )
Cleanup the main test ( )
add multithreaded compressed streams tests
Docs : Fix typo in scripting . asciidoc
[ ML ] [ Inference ] adding tree model ( )
Docs : Use the default analyzer
Throw parsing error if fuzzy query contains multiple fields
MINOR : Remove SNAPSHOT Suffixes in Gradle ( )
Merge pull request from javanna / enhancement / terms_lookup_query_builder
Mute RecoverySourceHandlerTests . testCancelRecoveryDuringPhase1
Docs : Fix typo in RestIT
added missing {
Merge pull request from nik9000 / painless_generated_skip_checkstyle
Improve recovery of applyDeletedShards
[ STORE ] Calculate Alder32 Checksums for legacy files in Store # checkIntegrity
Fix P / C assertions for rewrite reader
Remove DiscoveryNodesProvider interface ( )
Merge pull request elastic / elasticsearch from rjernst / move_license
better date
Remove mention of tribe filesystems
[ DOCS ] Re - add include - xpack attribute
remove unused import
Basic derivative reducer
make commitID similar to search queries , closes .
Update to Lucene 4 . 10 . 2
removed dead code
[ DOCS ] Add data streams to flush API docs ( )
[ DOCS ] Add redirect for painless examples anchor
Finish removing loadConfigSettings , and fix node startup to use correct
Bump BWC version on bad allocation request test
Allow to Debian REST tests when master ( )
Merge pull request from abeyad / feature - suggest - refactoring
update to 0 . 12 . 0 SNAPSHOT
Merge pull request elastic / elasticsearch from rjernst / move_license
prepare release elasticsearch - cloud - aws - 2 . 0 . 0 . RC1
Merge pull request from rmuir / sane_permissions
Plugins : Simplified repository api for snapshot / restore
Docs : Add simple link to the Bulk policy status ( )
Merge pull request elastic / elasticsearch from rjernst / deguice1
Fix AwarenessAllocationIT . testAwarenessZones test
Mute failing test
[ DOCS ] Moves PKI realm file ( )
painless : Fix issue with dup and cast
Add support for write index in WildcardExpressionResolver ( )
[ ML ] Use PersistentTasksService # waitForPersistentTaskStatus ( . . . ) to wait for job and datafeed status and
Do not resolve addresses in remote connection info ( )
Remove dead code
Disable java and maven version checking
Merge pull request from rjernst / normalize_plugin_zip
Remove deprecated PutIndexTemplateRequestBuilder # setTemplate ( )
[ DOCS ] Add docs for ` geo_match ` docs ( )
Fix roles of search requests ( )
remove dead code
[ DOCS ] Add docs for ` geo_match ` token filter ( )
[ DOCS ] Fix link to docker content
Merge pull request from rjernst / fix / field - type - ref
Monitoring : Add MonitoringClientProxy
[ TEST ] fixes line length issue in BulkRequestModifierTests
Remove TEST interface in SnapshotShardsService ( )
[ DOCS ] Adds shared configuration section
removed javax . mail from test dependencies
[ Transform ] move ClusterDiscoveryConfiguration
[ DOCS ] Fixed typos
Remove ` RoutingAllocation . Result ` ( )
Merge pull request from rjernst / ingest_plugin_api
Merge pull request from jpountz / remove / multi_field
Merge pull request from jasontedor / feature / improve - java - version - comparison
Merge pull request from rjernst / deprecate_exception_hurter
Update to elasticsearch 1 . 3 . 0
Fix CoordinatorTests permissions
Remove legacy BWC test infrastructure and tests ( )
[ BUILD ] Use SuppressFrobidden annotation in bin / elasticsearch - plugin
Merge pull request from rmuir / jail_filestore_exception
Consolidate allocation name in QueryShardContext
Cherry pick tests and seqNo recovery hardning from
[ TEST ] Fix indexed script test .
Update README . asciidoc
Test : Fix mistake , this test requires cluster scope test
Fixes compile errors due to upstream changes in elasticsearch itself
[ TEST ] Fixed skip test for / / / allocated
Merge pull request from nik9000 / xlint_plugins
[ DOCS ] Fixed hardcoded link to unfreeze_index
initial work on local gateway
Remove AggregationContext . cacheRecycler ( ) .
[ DOCS ] Add redirect for ` geo_match ` token filter ( )
Cleanups for Def
Merge pull request from jdconrad / stype
Merge pull request from tebriel / plugin_docs
Remove generics from ExtendedBounds
[ DOCS ] add system comma
remove cruft
[ TEST ] wait for yellow else assertSearchResponse will trip
[ ML ] [ Inference ] changing setting to be memorySizeSettting ( elastic / x - pack - elasticsearch )
[ DOCS ] Add data streams to docs for docs ( )
Remove IndexRequestBuilder . setType ( ) ( )
[ TEST ] wait for yellow else assertSearchResponse will trip
[ TEST ] Add retries to MockRepository getRepositoryData to try to
Merge pull request from nik9000 / painless_generated_skip_checkstyle
remove old code
[ TEST ] Fixed delete query test issues
Merge pull request from johtani / cleanup_analyze_api_test
Local Gateway : Store specific points
Fix Infinite Loops in ExceptionsHelper # unwrap ( )
Remove references to mapping docs
[ DOCS ] Fixed broken link
Remove unused modifier
Merge pull request from ywelsch / fix / compressedstring_equals
less streams more loops
Remove now that yaml test plugins ( )
protected when not of the same type mapper when merging
Merge pull request from jdconrad / stype
Merge pull request from javanna / enhancement / client_ingest_methods
[ DOCS ] Added multi - index open / close
Internal : extend refresh - mapping logic to the _default_ type
Fixed typo
ILM Re - enable ID in tests
Recovery : allow to recover into a folder containing a corrupted shard
Add ability to split shards ( )
Add more ensureGreen to QueryShardContext tests
Merge pull request from rjernst / fix / field - type - merge
Fix InternalSearchHits serialization to be deterministic
Remove deprecated settings
[ ML ] [ Inference ] [ Monitoring ] Add necessary lang ident classes ( elastic / x - pack - elasticsearch )
[ TEST ] Add AwaitsFix to ensure try / ensure the transport response
minor chagne
Fix Infinite Loops in ExceptionsHelper # unwrap ( )
Mute ClusterDisruptionIT testAckedIndexing
lucene 4 : Upgraded ContentIndexSearcher
Mappings : Lock down _type field
[ TEST ] add support for binary responses to REST tests infra
Fix YAML for plugins ( )
Move owner to elastic
Fix yamlBuilder ( )
[ TEST ] Simplify SearchSourceBuilderTests setup
SQL : Fix incorrect field type for SYS TYPE ( )
update to elasticsearch 1 . 0 . 0
remove dead code
Merge pull request elastic / elasticsearch from rjernst / rest_alone
[ DOCS ] Added missing section include API ( elastic / x - pack - elasticsearch )
[ TEST ] Fix CommonTermsQueryBuilderTest
Adjust version skips for intervals stats
Add support for ` authorization_realms `
Remove XContentParser # maxOpenFiles
300_sequence_numbers should not rely on 7 . 0 total hits structure
[ TEST ] fixing ClusterApplierService
Merge pull request from rmuir / more_indy_typing
Merge pull request from tebriel / plugin_docs
[ TEST ] Randomize FS Translog implementation
[ DOCS ] Reformat ` kstem ` token filter ( )
Merge pull request from rjernst / normalize_plugin_zip
upgrade to lucene 5 snapshot
Merge pull request from rmuir / sane_permissions
[ TEST ] Fix explain - ingest - query and tests ( )
[ DOCS ] Changes failed transforms limitation . ( )
[ DOCS ] Add few icons to stack - tools . asciidoc ( )
Merge pull request from ywelsch / fix / index - settings - restore
upgrade to lucene - 8 . 7 . 0 - snapshot - b313618cc1d ( )
[ DOCS ] Add abbreviated titles
[ CCR ] Remove unused Request constructors ( )
[ ML ] [ Inference ] protecting doSaveState with optimistic concurrency ( )
clean code
[ TEST ] fix CommonTermsQueryBuilderTest
TestConsumerPlugin shouldn', "'", 't register its service on transport clients
Compilation : Fix compilation issue with latest 1 . 4 - SNAPSHOT
Added count percolate api
lucene 4 : Upgraded ContentIndexSearcher
Merge remote - tracking branch', "'", 'origin / master', "'", 'into json_strict_duplicate_checks
switch to client pom
[ DOCS ] Add data streams to flush API docs ( )
Merge pull request elastic / elasticsearch from dliappis / master
Remove @ Override : true to tests
Mute ESTestCaseTests # testBasePortGradle
checking on index metadata does not make sense for mappings . . .
Failure to update search with empty formats , closes .
Avoid copying searchable snapshots ( )
Fix logging statement
[ DOCS ] Add redirect for heap size ( )
[ DOCS ] Add redirect for heap size ( )
Search : Search requests hangs when no indices exists , closes .
Fix Infinite Loops in ExceptionsHelper # unwrap ( )
[ DOCS ] Fixed typos
allow to control the right field
Prevent MockStore where commits should be prevented on close
update to mvel 1 . 7 . 0
[ DOCS ] Adds dataframe transforms limitation . ( )
[ CCR ] Implement CompositeIndicesRequest
Fix StackOverflowException for long suggestion strings
Remove support for address ` delimited_payload_filter ` ( elastic / elasticsearch )
Remove unused import
Add a BridgePartition to be used by testAckedIndexing ( )
[ DOCS ] Fixed percolate docs errors
Watcher : Add proxy support .
Merge pull request from rjernst / fix / 12329
Query DSL : Allow to default certain settings in query_string / field names , closes .
[ DOCS ] Fix typos in create data frame API API ( )
lucene4 : finish org . elasticsearch . common . compress cleanup
Share types in SearchSortValuesTests ( )
Update Azure - plugin . asciidoc
Mute RecoverySourceHandlerTests . testCancelRecoveryDuringPhase1
[ ML ] Delete anomaly detection post data endpoint ( )
Fix CoordinatorTests bug ( )
[ DOCS ] Fix typos in EQL docs
Test : Fixed pluggable exceptions
Fixed typo
Query refactoring : query_string
Query DSL : Allow to default certain settings in query_string / field / type , closes .
Term Vectors : support for version and version_type
fix test bug
Docs : Fix typo in timestamp - field . asciidoc
Recovery : failed to properly ack translog ops during wait on mapping changes
[ TEST ] Add missing parts in delete_expired_data REST tests
[ DOCS ] Remove beta label from metricbeat documentation ( )
awaited fix a couple of index template tests till we have removed the ingest index template
[ TEST ] Left from IntervalSchedule .
update documentation with release 2 . 5 . 0
test : added missing of date math in the unified highlighter ( )
[ TEST ] Pass silence failing bwc test
[ DOCS ] Added xpack . watcher . node_concurrent_job_allocations version ( elastic / x - pack - elasticsearch )
[ TEST ] Fix CommonTermsQueryBuilderTest
[ DOCS ] Reformat create index API docs ( )
Merge remote - tracking branch', "'", 'upstream / master', "'
Fix Infinite Loops in ExceptionsHelper # unwrap ( )
[ DOCS ] Reformat create index API docs ( )
Cleanup ClusterServiceIT # testClusterStateBatchedUpdates
Remove deprecated PutIndexTemplateRequestBuilder # setTemplate ( )
fix test .
[ DOCS ] Add missing comma
Merge pull request elastic / elasticsearch from rjernst / license_state_removal
[ ML ] Use JobTaskStatus # staleTask ( . . . ) instead of PersistentTask # needsReassignement ( . . . ) for checking whether a job task is stale when allocation a datafeed to a node .
[ DOCS ] Fix link to X - Pack Kibana ( )
Update skip after backport ( )
Test : Fixed pluggable exceptions
[ TEST ] Mute TimeoutCheckerTests . testWatchdog
[ DOCS ] EQL : Document ` between ` token filter ( )
Java api : remove ` kstem ` and ` _index ` , closes .
[ Docs ] Correct minor typo in templates . asciidoc ( )
[ ML ] Add created_by info to usage bwc tests
only ask for the relevant stats
javadoc
add check at end of each test to see if documents are the same
nicer logging
Remove unused printStackTrace
Add fromXContent method to SuggestBuilder
Merge pull request from rmuir / nukeSettings
Mute failing test ( )
Remove Redundandant
Rename IndicesAdminClient # existsAliases to IndicesAdminClient # aliasesExist .
Merge pull request from rmuir / harden_integration_tests
Added the API definition for deleting the index
Remove type from QL', "'", 's field_caps requests ( )
Merge pull request from rmuir / rat
remove bashism in bin / elasticsearch
Renamed allowNoIndices to failNoIndices and easier in order to setters .
[ UTIL ] Moved the JSON spec generator into the ` utils ` folder
Warmers : Have an explicit warmer thread pool
Add fromXContent method to HighlightBuilder
[ DOCS ] Fix ` search . max_buckets ` token filter docs ( )
[ DOCS ] Fixed link to breaking changes
Merge pull request from rjernst / maven_publish
Document commit
Merge pull request from rjernst / dont_hide_log4j_exceptions
allow to control the buffer size of the gateway blob container
Update maven assembly plugin to latest version : 2 . 3
Also lock if a exception is the same
addressed review comments
[ ML ] Reuse encapsulated delete - by - query request
[ DOCS ] Streamline GS search topic . ( )
Merge pull request from djschny / issue10371
[ DOCS ] Fix typo ( elastic / x - pack - elasticsearch )
Remove legacy BWC test infrastructure and tests ( )
Fix inconsistency in synonym docs ( )
Update to elasticsearch 1 . 0 . 0
Use CacheService Persisted Cache Size ( )
Correct id parametrization in deciders ( )
Add SAML IdP plugin for internal use ( )
Add migration docs for 8 . 0 beta2
Merge pull request from ywelsch / fix / compressedstring_equals
Cut global rest test plugin to use expectThrows ( )
Query refactoring : query_string
[ Docs ] Update length - tokenfilter . asciidoc ( )
Add NioTransport threads to thread name checks ( )
Merge pull request from rmuir / spi_reload
apply review comments
[ DOCS ] Fix typo in RoutingNode docs ( )
[ ML ] Change the root_cause of error responses to be more informative ( elastic / x - pack - elasticsearch )
Fix default DeleteIndex Timeout
[ DOCS ] Adding index . txt
remove the settings_require_units setting .
provide version in the english filter
Made ` index ` in TermsLookup mandatory ( RestStatus . CONFLICT ) , closes .
SQL : Fix incorrect field type in SYS TYPE ( )
Merge pull request from rjernst / fix / 12329
[ ML ] adds new feature_processors field names ( )
move to disable', ['Marvel
Upgrade to joda 2 . 0 . 2
[ TEST ] Fix test bug in MultiOrdinalsTests .
Remove AggregationContext . cacheRecycler ( ) .
Fix yamlBuilder
always return , closes .
[ TEST ] Clean up watch created by stats test ( )
Remove usage of tuple as a method parameter
[ ML ] [ Data Frame ] make sure that fields exist when creating progress ( )
rename java 9 to elasticsearch )
Query DSL : ` constant_score ` via query_string , closes .
Remove legacy BWC test infrastructure and tests ( )
Don', "'", 't throw IllegalInstantException to determine DST gap
Mget is no longer enabled
Emit settings deprecation logging on empty update
Recovery : failed to properly ack translog ops during wait on mapping changes
Scripting : Make _score in groovy scripts comparable
Term Vectors
Inline NettyHttpChannel # resolveIndexRouting
Fix TcpTransport . testRelocatedClosedIndexIssue ( )
Merge pull request from rjernst / docs / 17412
Add Netty ByteBuf Leak Check to REST Test Clusters ( )
Add brackets to split REST tests
Merge pull request from rmuir / jail_filestore_exception
If a order of InnerHitBuilder to 403
Add integration test
Pure ( )
[ DOCS ] Add data streams to flush API docs ( )
remove BufferedDeletesStream
Remove LocalTransport in ExtendedBounds
Merge pull request from spinscale / 1503 - cleanup - remove - bytesarray - unsafe
Remove Obsolete BwC Serialization Code ( )
prepare for next development iteration
Merge pull request from abeyad / feature - suggest - refactoring
[ TEST ] Mute IndicesCustomDataPathTests
[ TEST ] Revert accidential massive iterations
prepare release elasticsearch - analysis - stempel - 1 . 8 . 0
update documentation with release 2 . 5 . 0
[ DOCS ] Add docs for ` geo_match ` token filter ( )
Cherry pick tests and seqNo recovery hardning from
Tests : Force assertSameSyncIdSameDocs
add field for version to the explain API
[ DOCS ] Fixes broken links in reindex docs ( )
Remove PROTOTYPE from ExtendedBounds
Remove Obsolete BwC Serialization ( )
Make groovy sandbox method blacklist dynamically additive
Log MockStore in the cluster state rest api
[ DOCS ] Add missing icons to transform HLRC APIs ( )
Test : ClusterServiceTetsts . testSegmentsWithMergeFlag should close it', "'", 's translog
Add wabisabi to Scala clients and clients ( )
Correct GoogleCloudStorageHttpHandler ( )
Remove deprecated PutIndexTemplateRequestBuilder # setTemplate ( )
Fix inconsistency in synonym docs ( )
Added version 2 . 3 . 5 with bwc indices
Fix CreateRepository Requeset in HLRC ( )
Security : Adjusing assertion to be enabled
Fix HttpClientTests ( elastic / elasticsearch )
Fix comment grammar
[ ML ] [ Inference ] changing setting to be memorySizeSettting ( )
trimmed down SimpleQueryTests + more assertAcked
Merge pull request elastic / elasticsearch from rjernst / license_state_removal
[ TEST ] Fix incorrect location in createIndicesWithRandomAliases
If a order of InnerHitBuilder to 403
painless : Clean up PainlessMethod ( )
Fix Creating NOOP Tasks on SNAPSHOT Pool ( )
protected for , closes .
[ TEST ] Randomize FS Translog implementation
Remove Redundant Documentation on SnapshotsService ( )
[ TEST ] Don', "'", 't expect acks when isolating nodes
[ TEST ] Use array * helper assert methods instead of verifying the array directly
Add missing build - plugin dependency
Merge pull request from dongjoon / hyun / rename_variables
[ TEST ] Enhance failure message when bulk updates have failures
Move provided lucene - expression jar below the test - framework
clean code
[ TEST ] HttpCompressionIT is a real IT
[ DOCS ] Fixed typos
minor cleanups
Merge pull request elastic / elasticsearch from rjernst / search_parser
clean code
Prepare 1 . x branch
Merge pull request from rjernst / fix / field - type - ref
Merge pull request from s1monw / remove_bloom
Merge pull request from nik9000 / xlint_plugins
[ ML ] move DataStreamDiagnostics into DataCountsReporter ( elastic / x - pack - elasticsearch )
Merge pull request from rmuir / rat
reenable bwc test now that is merged ( )
Move owner to elastic
Register MockFSDirectoryService settings
Reject invalid test logging annotations
Docs : Corrected synonym tokenfiler docs
Merge pull request elastic / elasticsearch from rjernst / move_license
Docs : Use task_id in examples of tasks ( )
INGEST : Add Pipeline Processor ( )
Add support for ` auto_date_histogram ` ( )
readonly changes for azure discovery types
Silence SplitIndexIT . testSplitIndexPrimaryTerm test failure . ( )
Merge pull request from rjernst / client_jar_id
SNAPSHOT + TESTS : Speed up Snapshot IT ( )
Update to Elasticsearch 0 . 90 . 6 / Lucene 4 . 5 . 1
Add Pause / Resume Auto ( )
Merge pull request from rjernst / fix / field - type - merge
Infer KeyStore type from pathname ( elastic / x - pack - elasticsearch )
[ DOCS ] Add redirect for ` geo_match ` token filter ( )
Fix compilation in Ecipse ( )
add a comment why and when we can ignore skipTranslogRecovery when serializing
Add a BridgePartition to be used by testAckedIndexing ( RestStatus . INTERNAL_SERVER_ERROR )
[ TEST ] Fix incorrect discovery options .
Remove unused code
Remove JMX connector creation flags , and JMX attributes
[ TEST ] use assertAcked when creating indices
Merge pull request from rmuir / sane_permissions
Merge pull request from rmuir / rat
reenable bwc test now that is merged ( )
[ DOCS ] Warn about searching across all fields wt . ` query_string ` ( )
Fix testRetentionLeasesClearedOnRestore ( )
Fix Translog # testWithRandomException ( )
Merge pull request from jdconrad / stype
Alerting : Add index action .
HLRC : add support for the searchable_snapshot ILM action ( )
Fold InitialSearchPhase into AbstractSearchAsyncAction ( )
Share wording in MappingQueryParser
HLRC : add support for the searchable_snapshot ILM action ( )
Painless : Complete Removal of Painless Type ( )
[ DOCS ] Add missing icons to docs ( )
[ ML ] Remove redundant logging
Remove unused User class from protocol ( )
[ DOCS ] Fix typos in context suggest
Revert " Integrates soft - deletes into Elasticsearch ( ) "
Merge pull request from rjernst / save_eclipse
Replace manual copying an array or collection with static methods calls ( )
Build : Only build BWC version to 5 . 6 ( )
[ TEST ] wait for yellow else assertSearchResponse will trip
Remove dead code
Avoid import
[ DOCS ] Add redirect for ` size ` query docs ( )
move to mvel 2 . 1 . beta1 ( versioning changed from 2 . 1 . rc1 )
Docs : Fixed heading link
Fixed typos in synonym docs ( )
Mute ESTestCaseTests . testCancelRecoveryDuringPhase1
Merge pull request elastic / elasticsearch from rjernst / license_state_removal
[ DOCS ] Move ` index . http . concurrent_small_file_streams ` setting ( )
Mute failing test
Merge pull request from rmuir / xlint_plugins
Document name is supported . asciidoc
Scripting : Make _score in groovy scripts comparable
Build : Enforce similar tasks for AD deps ( )
[ Monitoring ] Removing Dedicated IndicesStatsCollector ( elastic / elasticsearch )
Update links to release ( )
Fix testSeqNoCollision ( )
Update TESTING . asciidoc
[ DOCS ] Fix X - Pack tag for Asciidoctor ( )
Add Inference Pipeline aggregation to HLRC ( )
[ DOCS ] Add PIT to search after docs ( )
Fix casting for PreConfiguredTokenFilter ( )
Create Index API : Allow to provide mappings . closes .
Merge pull request from rjernst / ingest_plugin_api
Better Exception in NetworkUtilsTests ( )
switch to plugin pom
Fix Infinite Loops in ExceptionsHelper # unwrap ( )
[ DOCS ] Add PGSync as community - supported integration ( ) ( )
Merge pull request from rjernst / refactor / field - type - merge
[ DOCS ] Fix typos in RoutingNode docs ( )
Fix ` 408 REQUEST_TIMEOUT ` ( )
Create es - 1 . 2 branch
Formalize usage stats for analytics ( )
add missing imports
Adding support for write index names
Add start_basic task for 2 . 0 ( )
Tests : Fix systemkeytool tests
doc
[ DOCS ] Reformat ` porter_stem ` token filter ( )
[ DOCS ] Move explicit anchors for PKI realms ( )
Also removing InternalPolygonBuilder
[ ML ] Remove unused Request constructors ( )
Merge pull request from s1monw / remove_bloom
The field parameter of ` auto_date_histogram ` ) ( )
Watcher : Make SchedulerEngine watch handling threadsafe ( elastic / elasticsearch )
Clean up XContentBuilder in X - Pack
unmuted test
Docs : Use the default stemmer reference to the painless page .
[ ML ] Deprecate anomaly detection post data endpoint ( )
Enable the ingest base test .
Modify load average formats
More tests
Fix testAutoFollowing assertion ( )
Update to Jackson changes 2 . 0 . 0
[ TEST ] use assertAcked in LdapTestCase
Fix exception message
HLRC : Add parameters to stopRollupJob API ( )
[ ML ] [ Inference ] allowing per - model licensing ( )
[ STORE ] Don', "'", 't include indexShard if it has been removed before
Shard Stats : Allow to control mappings in cluster , closes .
[ DOCS ] Adds performance considerations section to transforms overview ( )
Clear Indices Cache API : Allow to clear indices cache , closes
Fix DeleteRequest / GetRequest ( )
Merge pull request elastic / elasticsearch from rjernst / move_license
Fix build of plugins . yml on Windows
Merge pull request from javanna / enhancement / terms_lookup_query_builder
Adapt InternalTestCluster to auto adjust ` minimum_master_nodes ` ( )
Unmute test
prepare release elasticsearch - cloud - aws - 1 . 0 . 0
Merge pull request from rmuir / reformat_bootstrap
[ TEST ] Fixed whitespace in Ruby template for docs
[ TEST ] fixed SimpleQueryTests # testDateRangeInQueryString to specify the mappings upfront and wait for green
Query DSL : ` constant_score ` should throw error on more than one filter
[ Monitoring ] Remove Dedicated IndicesStatsCollector ( elastic / x - pack - elasticsearch )
[ DOCS ] Add redirect for heap size ( )
[ TEST ] Don', "'", 't expect acks when isolating nodes
Painless : Add more jar ( )
Merge pull request from jpountz / enhancement / stricter_metadata_parsing
Merge pull request from nik9000 / java_8_for_package_tests
Add 7 . 0 . 0 - alpha5 release notes
Query refactoring : NotQueryBuilder and Parser
SCRIPTING : Move sort Context to its Own Class ( )
Docs : Use the new experimental annotation .
Remove unused clear ( IndexReader ) method from IndexFieldData
Add opening state to datafeed API
HLRC : split xpack request converters ( )
Minor changes to no_match_size highlight parameter and highlight tests ( )
Add UBI docker builds ( )
SQL : Don', "'", 't allow inexact fields for MIN / MAX ( )
Fix P / GetRequest ( )
better handling of concurrent transport address in BulkProcessorIT
[ DOCS ] Add multi - index open / close
Merge pull request from s1monw / allow_master_elect_extension
[ TEST ] Add a lot of forgotten try with resources to wrap ElasticsearchResponses
Deprecate the new thread pool ( )
Enforce translog access via engine ( )
[ ML ] Fix model snapshot YAML tests
Fixed tests for info , ping and index
TESTS : Use File Based Discovery in REST Tests ( )
Fix compilation in SecurityNetty4HttpServerTransport
Docs : make the welcome page more obvious
[ TEST ] improve HLRC Job ( )
[ DOCS ] Fixed link to rollup content
Useful error message for null property placeholder
Merge pull request from rjernst / deprecate_exception_hurter
Docs : Fixed bad link
[ TEST ] Check accounting breaker is equal to segment stats rather than 0
Use default forceAnalyzeQueryString if no query builder is present
[ Rollup ] Don', "'", 't put and MlRestTestStateCleaner
[ Monitoring ] Add monitoring privilege ( elastic / x - pack - elasticsearch )
test : add missing date math to blacklist
[ TEST ] Revert " Increase " location " files "
Remove deprecated PutIndexTemplateRequestBuilder # setTemplate ( RestStatus .', "'", ') method
HLRC : Add ability to stopRollupJob API ( )
Added support for acknowledgements to cluster reroute api
[ DOCS ] Fix edge n - gram tokenizer nav
[ ENGINE ] Prevent NPE if engine is closed while version map is checked
Add ml licence plugin ( )
[ DOCS ] - Updating chunk_size values to fix size value notation . Chunksize41591 ( )
[ TEST ] fix skip test for wait_for_active_shards issues
Fix TimeSeriesLifecycleActionsIT . testRolloverAlreadyExists ( )
[ ML ] [ Inference ] changing setting to stop blackhole test ( )
Fix testAutoExpandIndicesDuringRollingUpgrade ( )
Add API ( )
Upgrade to Lucene 8 . 6 . 0 - snapshot - 08b8d116f8f ( )
Remove buildResources from BuildPlugin .
Remove LocalTransport in ExtendedBounds
Support SpanMultiTerm , closes ,
Fix guide
[ Tests ] Harden InternalExtendedStatsTests ( )
Remove unused User constructor ( )
Mute WatcherRestIT 20_minimal_body
apply review comments
Build : Fix issue with dup / directories
Merge pull request from rjernst / ingest_plugin_api
Merge pull request from rmuir / crazy_ubuntu
XContent : Factor non - groovy support and allow to 5 . 0 . 0 - SNAPSHOT
Monitoring : Hide Pipeline Based ( elastic / elasticsearch )
Fix compilation error
Docs : Fix typo in RestIT
Fix SuggestSearchTests ( )
Restructure the GS topics & add redirects . ( )
Cherry pick tests and seqNo recovery hardning from
Remove SecurityTransportModule
[ SPEC ] Fixed bad name in ping . json
Remove ParseFieldMatcher usages from SearchService
[ DOCS ] Fixed typos
move to 1 . 0 . 0 snap
prepare for next development iteration
Move ` client ` token filter ( )
[ ML ] Remove unused Request constructors ( )
Sorting on _score in the URI format is reversed , closes .
Revert " Mute failing tests in bin / elasticsearch ( )
Merge remote - tracking branch', "'", 'origin / master', "'
Bulk API : Add ` min_value_as_string ` and ` reverse_term ` ( )
Search : Search requests hangs when no indices exists , closes .
[ DOCS ] Merges duplicate pages for Active Directory realms ( )
[ TEST ] packaging client - by environment
Fix LoggingOutputStream to work on windows ( )
[ DOCS ] Documented gateway . local . auto_import_dangled
[ DOCS ] Fix broken link to unfreeze_index
Plugins : Simplified repository api for snapshot / restore
Refactor RestExecuteWatchActionTest w / Writeable . plugin
test : don', "'", 't randomly wrap index reader
Merge pull request from rmuir / shave_mustache
Fix get test
Merge pull request from rjernst / unused
Fixed compile errors due to upstream changes .
Add ( )
Merge pull request from ywelsch / fix / update - cs - with - routingresult
Update skip version for 5 . 0 . 0 - Beta1
update documentation with release 2 . 2 . 1
[ TEST ] Add SuppressSysoutChecks to DistributorDirectoryTest
Reenable Bwc Tests After ( )
[ DOCS ] Fixed hardcoded link to ES ref .
Add support for write index resolution to SearchRequest ( )
[ ML ] Remove dead delete job index code
Merge pull request elastic / elasticsearch from rjernst / move_license
Renamed allowNoIndices to failNoIndices and changed parameter order .
[ DOCS ] Add links to release notes and highlights
check can be loaded by wildcards .
Upgrade to Gradle 1 . 0 . 1
Update to AWS SDK 1 . 7 . 13
only ask for the relevant stats
Reindex : Do not write dummy documents in ExistsMissingTests ( )
Merge pull request from rmuir / painless_compile_exceptions
Tests : Mute IndexStatsIT
Fix build - to statement
Add migration check for deprecated ` delimited_payload_filter ` ( elastic / x - pack - elasticsearch )
test : add debug logging and cleanup stopping of servers
[ Docs ] Unify spelling of Elasticsearch ( )
geo_point runtime field implementation ( )
Merge pull request from rmuir / sane_permissions
add FailedToCommitException to registration
Add validation about versions for objectCount ( )
Marvel : Mute IndexStatsIT
Document documentation documentation in api docs
[ ML ] Do not skip rows with missing values for regression ( )
Mute failing test .
SQL : Add toXcontent test for SqlResponse ( elastic / x - pack - elasticsearch )
Docs : Fix typo in scripting . asciidoc
Remove legacy MetaDataStateFormat ( )
Revert javadoc change : iff is correct
Merge pull request from rmuir / painless_compile_exceptions
[ DOCS ] Add dataframe authorization section ( )
Support parameters when cleaning on java ( )
[ Docs ] Correct typo in removal_of_types . asciidoc ( )
Test : Added a test that verifies that we fail with 403 when the user doesn', "'", 't have the ` manage_watcher ` role .
SQL : Fix bug and added integ test runner
fix build
Merge pull request from nik9000 / painless_generated_skip_checkstyle
Fix underscore Requeset in HLRC ( )
Docs : Fixed bad link
[ TEST ] move testBulkRequestModifier to existing BulkRequestModifierTests class
painless : make compound statement like a type type .
Merge pull request from rmuir / rat
Add a more descriptive example to the Template
Enable testing for JDK 11 ( )
Do not execute ML CRUD actions when upgrade mode is enabled ( )
Include nested aggregation docs
[ DOCS ] Fix typo in RoutingNode comment ( elastic / x - pack - elasticsearch )
Mute failing test
Merge pull request from rmuir / rat
[ ML ] [ Inference ] protecting doSaveState with optimistic concurrency ( )
Improved filtering by _parent field
Query refactoring : query_string
[ DOCS ] Fixed typos
Fix packaging
Recovery : only cancel when primary completed relocation
Merge pull request from rjernst / integ_test_deps
TEST : Mute IndexStatsIT
Add missing ( )
Fix docs to run with install nodes
[ TEST ] increase timeout slightly in wait_for_active_shards test to allow
[ DOCS ] Add PKI realm configuration details ( )
Fix test bug
Remove deprecated PutIndexTemplateRequestBuilder # setTemplate ( )
Mute failing test
Watcher : Add MonitoringClientProxy
[ ML ] Remove record_count .
Docs : A GS results docs
Remove unused User constructor ( )
Text Query : add ` reverse_cout ` and ` max_value_as_string ` , closes .
Added support for acknowledgements to cluster reroute api
[ DOCS ] Add ML configuration details ( elastic / x - pack - elasticsearch )
[ DOCS ] Fix edge n - gram tokenizer nav
rename client - groovy to lang - groovy
AsyncSearchMaintenanceService should stop when closing a node ( )
Merge pull request from nik9000 / xlint_plugins
cleanup tests
[ Docs ] Update length - tokenfilter . asciidoc ( )
make a version master * , closes .
[ DOCS ] Reformat ` porter_stem ` enrich policy API ( )
Remove dead code
Upgrade to latest Lucene 2 . 2 . 2 , closes .
[ Infra ] Mute IndicesCustomDataPathTests
Fix typo in keyword . asciidoc ( )
prepare release elasticsearch - analysis - stempel - 1 . 11 . 0
SNAPSHOT : Add missing boost
Merge pull request from nik9000 / grep_minus_e_portability_fix
added install instruction with apt
Mute WatcherRestIT 20_minimal_body
Fix ` InternalEngineTests `
Move owner to elastic
Fix inconsistency in synonym docs ( )
update documentation after backport of ( )
Recycler : testTimeoutCleanup useful error if null is set to load ( elastic / elasticsearch )
[ TEST ] fix CommonTermsQueryBuilderTest
Fix test bug
fix perf bug with xor promotion
Merge pull request from jpountz / fix / compressedstring_equals
INGEST : Remove KV Processor ( )
Merge pull request from nik9000 / xlint2
review cleanups
Merge pull request from dadoonet / master
[ DOCS ] Fix typo : NamedID - > NameID ( )
fix typo in docs
Remove type _boost
Marvel : add base test tests
Data stream support for migrations deprecations info ( )
Respect lowercase_expanded_terms in MappingQueryParser
Mappings : Remove type prefix support from field names in queries
Merge pull request from rjernst / maven_publish
[ TEST ] Randomize FS Translog implementation
[ DOCS ] Add missing icons to transform HLRC APIs ( )
Fix line length
Limit request size on HTTP level
Fix BulkProcessor Retry ITs ( )
Mapper : Allow to control the ` _routing ` mapping , closes .
Add bwc support for pre - 5 . 0 completion index
[ DOCS ] document replacement for search exists
change missing filter serialization to a tranform result
Merge pull request from rmuir / shave_mustache
Add a BridgePartition to be used by testAckedIndexing ( )
Remove DiscoveryNode # simpleMatchToFullName in favour of existing DiscoveryNode # getLocalNode
[ TEST ] Fix test failure in CompletionSuggesterBuilderTests
Docs : Fix typo in RestIT
Merge pull request from rjernst / maven_publish
Merge pull request from s1monw / fold_in_index_cache_module
Update TESTING . asciidoc
300_sequence_numbers should not fail on ` setMinCompetitiveScore ` . ( )
Merge pull request from nik9000 / xlint2
Raised IOException on deleteBlob ( )
Remove Redundandant
[ DOCS ] Fix link to heap management APIs
Docs : Fix typo in RestIT
[ DOCS ] Fix ` search . max_buckets ` default ( )
Merge pull request from rjernst / deprecate_exception_hurter
Fix testAutoExpandIndicesDuringRollingUpgrade ( )
Authentication is not enabled
Add " locale " to true .
Merge pull request from nik9000 / xlint_plugins
Sorting on soft - deletes , closes .
Fix Infinite Loops in ExceptionsHelper # unwrap ( )
Fix more information . to test .
Fix line length
Merge pull request from rmuir / reformat_bootstrap
Fix CorruptedBlobStoreRepository Test ( )
Handle properly indexing rectangles that crosses the dateline ( )
Query DSL : Allow to default certain settings in query_string / field / type , closes .
Add ingest - attachment to official plugins
Add support for V2 index templates to SearchRequest API
Fix SuggestSearchTests to expect any order in the error message
Merge pull request from rmuir / spi_reload
Search : When searching against a type with a dfs type , dfs is ignored , closes .
Merge pull request from jimferenczi / copy_to_dynamic_object
Add missing logging param
[ ML ] Add the ML datafeed to the model config ( )
upgrade to lucene 8 . 0 . 0 - snapshot - 08b8d116f8f ( )
[ TEST ] Added SuppressSysoutChecks to DistributorDirectoryTest
[ TEST ] relax size test , to not run into OOM
Fix Infinite Loops in ExceptionsHelper # unwrap ( )
Merge pull request from rjernst / fix / field - type - merge
Merge pull request from s1monw / fold_in_index_cache_module
Fix BWC test ( )
[ DOCS ] Add PIT to search after docs ( )
[ ML ] [ Inference ] Add necessary to HLRC ( )
Add SLOPPY_ARC to GeoDistanceSearchBenchmark .
Fix Fuziness typo in multimatch error message and associated tests
Merge pull request from rjernst / jigsaw
Build : Fix subdirectories in Http Publish Addr to True ( )
Deprecate ` . options ` to init follower ( )
fix tests
Fix failure in RangeAggregator
Reenable bwc tests after backport of ( )
Merge pull request from s1monw / issues / 17412
[ ML ] Include message in field_stats for text log files ( )
Fix TcpTransport . testRelocatedClosedIndexIssue ( )
Merge pull request elastic / elasticsearch from rjernst / move_license
Remove benchmark privileges and actions from codebase and docs
Assert that do not guard against already failed engine
[ ENGINE ] Calculate Alder32 Checksums for legacy files in Store # checkIntegrity
Fix SearchableSnapshotActionIT ( )
Query DSL : Allow to default certain settings in query_string / field queries , closes .
fix test plugin visibility
Fix azure config checks ( )
ingest : Remove', "'", 'from', "'", 'type in docs
even on failure , don', "'", 't return a body for HEAD request
Adjust docs for Open / Close API ( )
Catch NoSuchDirectoryException on consistency check - the directory might not be there anymore
fixed compile error due to upstream change
Upgrade to lucene - 7 . 0 . 0 - snapshot - a0aef2f
[ ML ] Remove dead delete job index code
Fix yamlBuilder ( ) to return YAML builder instead of SMILE
[ DOCS ] Fix typos ( ) ( )
Fixed sample code for minhash ( )
Revert " Mute MinimumMasterNodesIT . testThreeNodesNoMasterBlock "
Merge pull request from rmuir / xlint_plugins
Merge remote - tracking branch', "'", 'origin / master', "'
[ CCR ] Add debug info tests for exceptions ( )
Adapt BWC for pre - deletes version after backport
add favicon
[ Painless ] Add boxed type to boxed type casts for method / return ( )
Merge remote - tracking branch', "'", 'es / master', "'", 'into enrich
fix test bug
Mute failing test
Merge pull request from rjernst / fix / 12329
Index creation and setting update may not return deprecation logging ( )
Make maven quietly
clean code
Remove Redundandant
Add non inspection checks
Aligned Transform . Result with all other result constructs
Merge pull request from jdconrad / cast
Remove name from version type .
Enable integ tests
Shorten read tests when running on plugins
Failure to execute search request with empty top level filter
Merge pull request from rjernst / dont_hide_log4j_exceptions
Merge pull request from nik9000 / painless_generated_skip_checkstyle
[ DOCS ] Add note about long - lived idle connections ( )
Added multithreaded getters .
Fix spelling
Use a separate build directory for Eclipse .
Move owner to elastic
Remove dead code
Merge pull request from jpountz / upgrade / lucene - 5 . 2 . 0 - snapshot - 1673124
change to 0 . 19 . 0 snap
Allow abort wrappers to a authN ( elastic / elasticsearch )
Aligned Transform . Result with all other result constructs
[ DOCS ] Fix typo in update . asciidoc
Docs : Use task_id in examples of tasks ( )
SQL : Add catalog and type arguments to SYS TABLES ( )
Remove deprecated RealmConfig constructor ( )
Query DSL : ` constant_score ` should throw error on more than one filter
Generate settings links for SEO
Mute failing test
Docs : Note that include_in_parent / root does not apply to geo - shape fields
[ TEST ] Fix CommonTermsQueryBuilderTest
[ TEST ] fix apache ds cleanup issues
Disable the pom . xml
remove flushing before running checkindex
Rename certgen
Doc : fix sliced typos ( )
Remove redundant modifier
Add fromXContent method to SuggestBuilder
Merge pull request from jdconrad / stype
[ DOCS ] Adds missing icons to transform HLRC APIs ( )
[ Tests ] Fix test failure in CompletionSuggesterBuilderTests
Skip netty4 yaml test in FIPS mode ( )
Add CONSOLE to distro the docs
Fix yamlBuilder ( )
[ DOCS ] Tidied up cluster settings docs
[ ENGINE ] Remove reflection call to waitForMerges
update documentation with release 2 . 5 . 0
addressed review comments
Fixes Tests so build passes
[ ENGINE ] Add ClusterDiscoveryConfiguration
Adapt BWC version for clone API ( )
Fix docs for script . allow_expensive_queries ( )
[ TEST ] Fix test names in MultiOrdinalsTests .
upgrade to lucene - 8 . 7 . 0 - snapshot - b313618cc1d ( )
[ ML ] do not start stopping tasks on reassignment ( )
[ DOCS ] EQL : Add search / index speed tip for functions ( )
Mute failing test
Fix SuggestSearchTests to expect any order in the error message
Update getting - started . asciidoc ( )
Added support for ` authorization_realms `
o support for the groovy cluster config
Remove LocalTransport in ExtendedBounds
[ ML ] Remove redundant logging
Added missing close test
Fix checkstyle in RangeAggregator
lucene 4 : Converted most Mappers over to FieldType API
Fix transport client tests for preventing %
Update to Lucene 4 . 10 . 0
[ DOCS ] Add . NET client overview to docs ( )
Merge pull request from jpountz / remove / multi_field
Merge pull request from djschny / issue10371
[ ML ] Add created_by info to usage stats ( elastic / x - pack - elasticsearch )
Remove InternalAggregation . Type ( )
Limit request size on transport level
Added 1 . 3 . 9 - SNAPSHOT and 1 . 4 . 4 - SNAPSHOT versions
[ CCR ] Implement CompositeIndicesRequest
Mute failing test
Minor cleanups in SyncedFlushService
[ docs ] pedantry
Tighten assertions in BulkProcessorRetryIT
Merge pull request from jasontedor / joiner - be - gone
[ ML ] Increase wait for templates timeout in tests ( )
[ DOCS ] " discovery . node . auto_import_dangled
[ ML ] [ Inference ] Add tree model ( )
[ reindex ] Implement CompositeIndicesRequest
Add support for ` es_port ` ( )
Docs : Update thrift . asciidoc
[ DOCS ] Warn about searching across all fields wt . ` query_string ` ( )
Merge pull request from s1monw / serialize_interrupted_exceptions
Watcher : Ignore HipChatServiceTests
Adjust bwc for recovery request ( )
Add / _cat / master .
Fix build build for test run ( )
[ ML ] Remove redundant logging
Allow users to be blocked after primary promotion ( )
Fix typo in comment in ReplicationOperation . java
Remove unused code
HLRC : Add Put Lifecycle Policy API to HLRC ( )
Merge pull request from gmarz / feature / virtuallock
[ DOCS ] Add missing icons to transform HLRC APIs ( )
[ Discovery ] fixed SimpleQueryTests # testDateRangeInQueryString to specify the mappings upfront and wait for green
[ DOCS ] Fixed link to docker content
Adressing review comments from IndexShard
Upgrade to Lucene 4 . 1 . 1
mute IndicesOptionsTests . testSerialization
Remove ImmutableMap $ Serialization ( )
remove dead code
[ TEST ] Fix CommonTermsQueryBuilderTest
Introduced missing imports
Add version 6 . 0 . 0 - rc2
Replace Streamable w / Writeable in SingleShardRequest and subclasses ( )
AllCircuitBreakerStats and CircuitBreakerStats to implement Writeable rather than Streamable
[ TEST ] Fixed elastic / x - pack - elasticsearch
[ ML ] Remove metadata constructors from data frame analytics BWC tests ( )
Add global checkpoint to use the breadth_first the type query
allow to [ 0 . 0 ] snap
Remove DiscoveryNodesProvider interface ( )
Merge pull request from rmuir / jdkhell
Merge pull request from jdconrad / stype
Merge pull request from s1monw / fold_in_index_cache_module
review cleanups
Fix handling of failed shard in LuceneChangesSnapshot ( )
[ ENGINE ] Move into much as possible into abstract Engine
[ TEST ] avoid double null check in TransportSearchActionTests
[ DOCS ] Fix typo in jump link . ( )
refactor breaking', "'", 's integration tests ( )
Make PercolatorFieldMapper parsing formats
Fix LoggingOutputStream to work on windows ( )
removed javax . mail from test dependencies
Add ILM poll_interval limit to breaking changes ( )
Mute failing test ( )
Add resiliency to Scala clients ( )
Silence failing test
Mappings : Lock down _type field
XFilteredQuery default strategy prefers query first in the deleted docs case
Fix SearchableSnapshotActionIT ( )
Add equals / hashCode to MainResponse ( elastic / elasticsearch )
fetching a shape should not occur on separate thread
fix constants
Add missing logging param
[ DOCS ] Update link to breaking changes
[ DOCS ] Add redirect for painless examples anchor
Fix readme
Merge pull request elastic / elasticsearch from rjernst / rest_alone
Fix Incorrect Transport Response Handler Type ( )
Remove ParseFieldMatcher usages from SearchService
Merge pull request from jimferenczi / close_tokenstream
Add build docker support
Compilation : Fix compilation issue with latest 1 . 4 - SNAPSHOT
Reenable tests
Fix P / C assertions for rewrite reader
Painless : Clean up PainlessMethod ( )
Mute CcrRetentionLeaseIT . testForgetFollower
Fix race in global checkpoint listeners test
auto_expand_replicas : Store CName , closes .
[ DOCS ] EQL : Add search / index speed tip for functions ( )
[ DOCS ] Add redirect for heap size ( )
Merge pull request elastic / elasticsearch from rjernst / move_license
Added the API definition for deleting the index
[ TEST ] add retries to MockRepository getRepositoryData to try to
XContent : Adapt to new method on parser ( )
Merge pull request from rjernst / fix / 12329
[ DOCS ] Add redirect for ` geo_match ` token filter ( )
Remove implicit PutIndexTemplateRequestBuilder ( )
Revert " Ensure domain_name setting for AD realm is present ( ) "
[ DOCS ] Reformat script query ( )
Fix CreateRepository Requeset in HLRC ( )
Merge pull request elastic / elasticsearch from rjernst / move_license
NPE in PluginManager when asking for list on non existing dir
[ DOCS ] Changes ML configuration pages to the explain docs ( )
Added support for acknowledgements to cluster reroute api
Mute failing SamlAuthenticationIT tests ( )
move to [ 2 . 0 . 0 - alpha5 ]
FlushNotAllowedEngineException from optimize
Fix testAutoExpandIndicesDuringRollingUpgrade ( )
Fix YAML test for watcher . allow_expensive_queries ( elastic / elasticsearch )
Search allocation awareness , closes .
Fix Creating NOOP ( )
Bump jackson dependency to a java package ( )
Switch AggregationSpec to ContextParser ( )
Fixed validate name name in docs
no support for objectCount , we can use keys
[ DOCS ] Add redirect for heap size ( )
Merge pull request from rjernst / fix / field - type - ref
[ Rollup ] Add support for setting settings
Add async_search Snapshot API ( )
[ ML ] [ Inference ] adding tree model ( )
Consolidate . gitignore entires for eclipse
[ DOCS ] Added SSL details details ( elastic / x - pack - elasticsearch )
Added the ability to set eager_global_ordinals in the new parent - join field .
Alerting : Split alerting into packages
Fully stop RetryableAction when cancelled ( )
[ TEST ] Renamed afterDistribution timeout to expectedTimeToHeal
Fix security manager with test copy docs
Merge pull request from mikemccand / close_tokenstream
Painless : Clean up PainlessMethod ( )
Rename a new thread pool ( )
Merge pull request from nik9000 / painless_generated_skip_checkstyle
Tribe node : add support for tribe node in shield
[ TEST ] use brackets
Better exception message
update version to 1 . 0 . 0
Simplify CcrRetentionLeaseIT . testForgetFollower
clean code
Index creation and setting update may not return deprecation logging ( )
Add debug logging for testRunningTasksCount
[ DOCS ] Add tab widget for TLS realm ( )
Merge pull request from rmuir / sane_permissions
Move Lucene', "'", 's LatLonShape
[ Transform ] Remove annotation
Fix compilation in Ecipse ( elastic / elasticsearch )
Merge pull request from rmuir / reformat_bootstrap
apply review comments
Fix docs for PreConfiguredTokenFilter ( )
Fix ` InternalEngineTests # testPruneAwayDeletedButRetainedIds `
Core : move TransportSingleCustomOperationAction in bin / elasticsearch
Bump 8 . 0 . 0 - alpha5 release notes
Update to elasticsearch 1 . 0 . 0 / Lucene 4 . 8 . 1
[ TEST ] make sure that IndicesRequestTests is repeateable using the same seed
update to 0 . 19 . 0
Fix DeleteRequest / GetRequest ( )
[ ML ] Mute data frame analytics BWC tests ( )
Add ` beforeIndexAddedToCluster ` callback
Remove Redundandant Loop in SnapshotShardsService ( )
test : don', "'", 't randomly wrap index reader
Merge pull request from rjernst / jigsaw
lucene 4 : Converted most Mappers over to FieldType API
Reenable Bwc Tests After ( )
[ ML ] Remove record_count from ML mappings ( elastic / x - pack - elasticsearch )
change rollover api to 400 in integration tests ( )
Fix InternalSearchHits serialization to be deterministic
improve geohash_filter to use terms filter
Remove dataCounts , NoneGatewayAllocator , & NoneGatewayModule
[ TEST ] Add missing parts in REST REST tests
geo_point runtime field implementation ( )
Add fromXContent method to SuggestBuilder
_all : Add missing boost
SQL : Doc on syntax ( identifiers in particular ) ( )
Merge pull request from rjernst / fix / 12329
Merge pull request from rjernst / ingest_plugin_api
Merge pull request from rjernst / fix / field - type - merge
Search : Add MonitoringClientProxy
Fix P / C assertions for rewrite reader
Mute DynamicIndexNameIntegrationTests ( )
Fixed bad URLs in docs
_all : Add missing boost
Merge pull request elastic / elasticsearch from dliappis / master
fix compile errors due to upstream changes .
[ Docs ] Correct typos in configuring - es . asciidoc ( )
Merge pull request from kkirsche / patch - 2
[ DOCS ] Fix anchor syntax
add support for auto_generate_synonyms_phrase_query in match_query for now
[ TEST ] Use a clientSSLContext in LdapTestCase
[ CORE ] Allow rebalancing primary shards on shared filesystems
ingest : Remove use of joda ( )
[ DOCS ] Fixed broken link
[ ML ] Do not skip rows with missing values for regression ( )
[ TEST ] Remove usage of @ SuppressForbidden ( )
Query refactoring : validate inner queries whenever supported
apply review comments from @ nik9000
Remove HostsSniffer in Sniffer
Upgrade to Lucene 4 . 2 . 1 ( )
don', "'", 't return default types .
Fix - config . boxes ( )
[ DOCS ] Add links to TLS links ( )
Merge pull request from jasontedor / strings - split - as - array
Add UBI docker builds ( )
[ CORE ] Don', "'", 't update indexShard if it has been removed before
SQL : Fix incorrect merging test ( )
Update to AWS SDK 1 . 7 . 13
FlushNotAllowedEngineException in PluginManager when asking on list on non existing dir
ingest : removed unused processor ( )
Fix ingest pipeline _simulate api with empty docs never returns a res … ( )
Remove SYS CATALOGS leftover
Remove DocumentFieldMappers ( )
[ DOCS ] Fiexes typo in getting started doc
Merge pull request from s1monw / serialize_interrupted_exceptions
[ ML ] [ Inference ] Unify top_classes object field names with analytics ( )
Upgrade to Lucene 4 . 1 . 1
Build : Fix typo in RestIT
Merge pull request from rmuir / more_indy_typing
add missing javadocs
[ CCR ] Remove unused IndicesStatsCollector ( )
Adapt to yaml test client upstream changes ( )
[ DOCS ] Added a few clarifications to the docs from the issues list
Merge pull request from s1monw / remove_bloom
Fix failing test on windows
Merge pull request from jasontedor / settings - cleanup
Stop disabling explicit GC
[ DOCS ] Reformats cat nodes API ( )
Fix test bug
Shard Allocation : Add ` min_value_as_string ` and ` max_value_as_string ` to ` _index ` keys , closes .
Add Pause / Resume Auto Follower API ( )
[ DOCS ] Reformat ` remove_duplicates ` token filter ( )
Docs : CONSOLEify synonym tokenfiler docs
Fixed docs for intervals . asciidoc ( ) ( )
Add cliSetup command to test clusters configuration ( )
use SecurityTransportModule
Watcher : Make SchedulerEngine job handling threadsafe ( elastic / x - pack - elasticsearch )
[ Docs ] Correct minor typos in templates . asciidoc ( )
doc
Watcher : Add proxy support for slack execution ( elastic / elasticsearch )
[ ML ] Add created_by info to usage stats ( )
[ Test ] Improve ID of multiparts
Wrong exception thrown when snapshot doesn', "'", 't exist
add support for acknowledgements to cluster reroute api
Fix StackOverflowException to expect the deviation in the request .
Merge pull request elastic / elasticsearch from rjernst / license_state_removal
Update docs for Open / Close ( )
Move licenses for unit test changes ( )
Merge pull request from kkirsche / patch - 2
[ DOCS ] Add data streams to flush docs ( )
Fixed compile errors after removal .
add close flag for RemoteClusterService
Add pre - configured “ lowercase ” normalizer ( )
[ DOCS ] Reformat enrich stats API ( )
Search : Allow to specify the proper field
[ ML ] Rename JobProvider to JobResultsProvider ( )
Fix UpdateMappingIntegrationIT test failures
[ DOCS ] Fine - tunes training_percent definition . ( )
[ TEST ] Fixed test to try in test for now
Upgrade to Lucene 4 . 6
Merge pull request from mikemccand / close_tokenstream
[ DOCS ] Reformat unique token filter docs ( )
[ DOCS ] Describe how to identify active ML model snapshot ( elastic / x - pack - elasticsearch )
apply review comments
[ TEST ] Add a lot of forgotten try with resources to wrap ElasticsearchResponses
[ DOCS ] various documentation
Revert " Build : Add name license to snapshots ( ) "
Add SLOPPY_ARC to GeoDistanceSearchBenchmark .
fix test
Merge pull request from rjernst / unused
Remove dead code
Added getters for fs . path , fs . mount and fs . dev in node stats api
[ DOCS ] Added link to Beats blacklist
Merge pull request from rjernst / normalize_plugin_zip
[ DOCS ] Fix anchor syntax
Changed the number of forks exception message
Fix InternalSearchHit # testForceStaleReplicaToBePromotedToPrimary ( )
Skip check for PreConfiguredTokenFilter ( )
Alerting : Remove getopt support in field .
fix rest boolean
Merge pull request from rjernst /', ['[ / field - type - type - merge
Merge pull request from mikemccand / close_tokenstream
[ DOCS ] Reformat enrich stats API ( )
Remove support for _all type in _analyze docs ( )
[ DOCS ] Fix agg block in multi anomaly detection ( )
add faker
Merge pull request from rmuir / jail_filestore_exception
Adjust wording according to review .
Update the API docs on 2 . 0 . 0
Remove types in BuildPlugin ( )
SQL : fall back to using the field name for column label ( )
Fix maven assembly warning about using root dir
Adapt changes in AcknowledgeResponse
Limit request size on HTTP level
Added sort support for the percolator query in the error message
[ TEST ] Left - over from 14420d7c4e15df9b565b50ef5beab797f756c3ac .
Fix compilation error
remove ParseFieldMatcher usages from SearchService
Tests : Mute IndexStatsIT
Fix exit conflicts
apply feedback from @ jaymode
Test : Fix pluggable mapper module to support transportclient
fix build
Fix test bug
Move owner to elastic
change rollover rest - api placeholder name
[ DOCS ] Chunk depth now configurable , so [ float ] not required
TransportMasterNodeOperationAction : retry operation if cluster state version changed while adding a ClusterStateListener
Mute failing test
Fix checkstyle error
[ Docs ] Clarify size parameter in Completion Suggester doc ( )
Merge pull request from rjernst / normalize_plugin_zip
Mute failing test
TransportInfo to implement Writeable rather than Streamable
update to - 3 . 2 . 0 / Elasticsearch 1 . 0 . 0
[ TEST ] fixing ClusterApplierService
Added PERL reserved words in REST keywords ( )
only ask for the relevant stats
Tribe node : add support for tribe node in shield
Extend customizing IOException tests ( )
[ ML ] Fix model snapshot YAML tests
Docs : Fixed broken link
[ ML ] Include message in field_stats for text migration files ( elastic / x - pack - elasticsearch )
Merge pull request from rjernst / module_culling3
Remove garbage
[ Tests ] Introduced ClusterDiscoveryConfiguration
SQL : Fix bug with optimization of null related conditionals ( )
[ DOCS ] EQL : Add search / index speed tip for functions ( )
[ Tests ] Harden InternalExtendedStatsTests ( )
[ DOCS ] Add painless section for painless examples ( )
Remove BWC BWC tests
[ DOCS ] Creates shared attributes for overview ( )
Fix build in test ( )
[ ML ] Remove record_count ( )
Local Gateway : Store , closes .
Re - enable bwc tests after was backported
Add more info in GeoBoundsAggregatorTests ( )
Made rest', "'", 's ` client ` ` ( )
Add ability to split shards ( )
Fix OriginationDate response
Make field type that are ( )
Cleanups for Def
Fix test bug
[ ML ] Reuse encapsulated delete - by - query request
[ ML ] Add a " ingest " parameter to inference blackhole ( elastic / x - pack - elasticsearch )
[ TEST ] Add missing skip in delete_expired_data REST tests
Merge pull request from javanna / enhancement / simulate_index_type_not_required
Internal : Simplify SecurityContext dependencies
Merge pull request from rjernst / fix / field - type - ref
add repo and license
Fix Checkstyle RedundantModifier
[ ML ] MlMemoryTracker should fail rows on reassignment ( elastic / x - pack - elasticsearch )
Minimum version check after backport
Update TESTING . asciidoc
Query DSL : ` constant_score ` should throw error on more than one filter
mute failing test
harden tests
[ Rollup ] add support for templates ( )
Watcher : Remove unnecessary Suffixes from query_string parser ( elastic / x - pack - elasticsearch )
Add facility to cross interface
Merge pull request from rjernst / dont_hide_log4j_exceptions
Painless : Correct ClassToName string conversion ( )
[ TEST ] Added missing script fields in REST tests
Merge pull request from jpountz / upgrade / lucene - 5 . 2 . 0 - snapshot - 1673124
Remove two test methods in favor of existing DiscoveryNode # getLocalNode
Change ClusterState # valueFetcher to more constructor
Docs : Add missing link to index . asciidoc
[ DOCS ] Add redirect for heap size ( )
[ ML ] [ HLRC ] Add necessary to regression ( )
Fix LoggingOutputStream to work on windows ( )
nicer logging
lucene4 : unit tests cleanup
Merge pull request from jdconrad / stype
switch to plugin pom
Grok : Add java action to the DeleteAlertResponse
Merge pull request from jasontedor / heap - size
[ DOCS ] Document ` indices . remote . scheme . scheme ` setting ( )
Merge pull request from jimferenczi / max_merge_settings
Wait for the license license tests
Recovery : failed to properly ack translog ops during wait on mapping changes
 )
Fix indentation in synonym docs ( )
Persistent Tasks : remove retries on notification failures ( )
return default DeleteIndex to use ParseFields
Merge pull request from rjernst / maven_publish
Merge pull request from rmuir / harden_integration_tests
Fix newline in testGetSnapshotsRequest ( )
Adapt BWC for 50836 ( )
[ DOCS ] Fixed hardcoded link to ES ref .
[ TEST ] Wait in rolling_upgrade rest test for monitoring indices ( )
[ DOCS ] Fix X - Pack tag for Asciidoctor ( )
Merge pull request from elastic / fix / indices - get - restspec
Merge pull request from rjernst / module_culling3
protected when not of the same type mapper when merging
Watcher : Ignore deactivated state validation ( elastic / elasticsearch )
[ TEST ] Fix test bug in MultiOrdinalsTests .
Merge pull request from rjernst / normalize_plugin_zip
reenable bwc test now that is merged ( )
Fix line length
Fix agg profiling when cancelled ( )
Deprecate the', "'", '_type', "'", 'in anchor .
fix bwc tests with empty indices in WildcardExpressionResolver .
[ doc ] readme
Painless : Add Pipeline Processor ( )
Merge pull request from rjernst / jigsaw
Fix uncaught checked exception in AzureTestUtils
[ TEST ] Fixed indentation in Ruby template for query generator
Merge pull request from rjernst / client_jar_id
[ DOCS ] add note about dynamic scriptings and updated links in the profiler
Merge pull request from rmuir / rat
[ ML ] Use PersistentTasksService # waitForPersistentTaskStatus ( . . . ) to wait for job and datafeed status and
Make PercolatorFieldMapper parsing less lenient .
Fix unit test
Fail if all inputstream within reindex
Merge pull request from rmuir / jail_filestore_exception
[ DOCS ] Fix typo in jump link . ( )
Painless : Complete Removal of Painless Type ( )
use the high transport channel for pings with transport client
Merge pull request from s1monw / cleanup_search_request
Merge pull request from rmuir / painless_compile_exceptions
[ ML ] Use PersistentTasksService # waitForPersistentTaskStatus ( . . . ) to wait for job and datafeed status and
Make consistent empty retention lease supplier
Fix SearchableSnapshotActionIT ( )
[ DOCS ] Reformat unique token filter docs ( )
Update docs for Open API ( )
[ DOCS ] Augmented token API description ( elastic / x - pack - elasticsearch )
Rest : Add missing parameters to indices . upgrade
Reinstate mutate docs to migrate_3_0 . asciidoc
Transport : A useful error message on potential HTTP connect
[ DOCS ] Fine - tunes training_percent definition . ( )
Fix mixed integ test .
[ DOCS ] Added explicit " lang " field to documentation of script score definition
[ TEST ] Fix test bugs in MultiOrdinalsTests .
Merge pull request from rjernst / maven_publish
FlushNotAllowedEngineException in PluginManager when asking for list on non existing dir
Update README . asciidoc
[ ML ] Change the root_cause of the datafeed ( )
Enable YAML REST test to use hardcoded ports

