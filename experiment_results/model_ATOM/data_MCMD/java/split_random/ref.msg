suppress Fortify false positive
Shard state action request logging
Add apply method to WebClient . Builder
Remove Record API dependencies from WordCount compiler test
Improve sanitization for list of URI types
[ FLINK - 4666 ] [ core ] Declare constants as final in ParameterTool
TraceRA : SSI construction does not need an LIRGeneratorTool .
KAFKA - 2993 : Calculate compression rate at close ( ) call
[ BAEL - 14846 ] - Fixed tests in spring - 5 - reactive module
Repeat / retry : fixed unbounded downstream requesting above Long . MAX_VALUE
Provide resolved information for new_local_repository
Do not disable TLSv1 . 3 by default
CSV export : config reading fix ( formatNumbers ) + cache settings .
core : ClassNode : Check is field really static or not in loadStaticValues ( )
Fixed code review comments .
[ Ml Data Frame ] Return bad_request on preview when config is invalid ( )
[ GR - 4606 ] Lookup Sulong libs in graalvm . home by default .
Simplify unordered floating point comparisons
SRC - 239 , thanks to a starter patch from Swapnil Raverkar
Improve connection reuse .
Fixed - RRateLimiter . delete ( ) method returns false .
Marvel : fix ShardsCollectorTests
[ TEST ] make BaseXContentTestCase platform independent ( bis )
[ JENKINS - 29956 ] Test Windows junctions before Java 7 symlink ( )
Saga SQLTransport support multiple sets of parameter for sql ;
[ FLINK - 1530 ] Cache deserialized ExecutionConfig in the AbstractInvokable
Verifying the Truffle / Java interop works OK also in asynchronous mode
Avoid gratuitous JDK 6 dependencies .
Extend the Scratch class to have a working directory .
Use long constants in code when they are long .
Fix treating all DVB data as AC - 3 .
refactor ShardingXADataSource .
Make isRunning protected .
Fix node field tests .
Improve exception messages for file output format when target file cannot be created .
Number format fix . Minimum fraction digits number depends on column settings .
refactoring
Allow path options to use user specific paths
MINOR : Restore and global consumers should never have group . instance . id ( )
GT - 3286 - Timestamp review fixes - final fixes
, fix test case
BAEL - 3693 : Rename test methods ( )
[ GR - 5221 ] Always scavenge the young generation , then optionally scavenge the old generation .
Enable default values for ( exceptional ) multiple options .
polish code style
Sulong : Improve error message when toplevel executable does not contain bitcode
Truffle - DSL : rename @ ImportGuards to @ ImportStatic
RPC server now deserializes to actual type of a call parameter , no longer the type given in the signature
Make soft - deletes settings final ( )
recover fatal error
Inline TransportSearchAction # doExecute
Fixes . Defaults for @ CacheNamespace do not match those of < cache >
for , refactor TableSegment ' s constructor
[ ROCKETMQ - 191 ] Fix socket options
am 16a39d79 : am 956db7bf : Merge " Fix AppBarLayout not being scrollable with fitSystemWindows = true " into mnc - ub - dev
Allow PATCH to have body / output
Add an option to allow local actions to run without locking outputs .
FB annotations and diamond in ReverseBuildTrigger . class
MINOR : missing fullstop in doc for ` max . partition . fetch . bytes `
[ hotfix ] [ tests ] Replace mockito - based verification with property verification .
Fix a bug with TTML using font size as % of cellResolution .
for : ignore create table if existed
fix unused argument
For , implement getPseudoColumns
Change TargetContext . findDirectPrerequisite from config1 = = config2 to config1 . equals ( c2 ) .
optimization
Allocation : add support for filtering by transport IP address
[ FLINK - 2919 ] Port FieldAccessMinibenchmark to JMH .
MySQL help provider error handle fix
Small fixes for ImageInfo API
improved sql parser .
Update Graal tutorial
[ hotfix ] Use semantically correct timeout in ExecutionGraph : : scheduleEager .
rename
SQL : Fix ODBC metadata for DATE & TIME data types ( )
Transparently handle migration from number to java . util . Duration
update unit test
[ java ] Fixing OS tests to work on Windows
[ FLINK - 15150 ] [ tests ] Prevent job from reaching terminal state
Improved " Discarded inbound message " warning ( )
CSV export : value quote fix
for Add channelHandlerContext for BackendConnection
[ GR - 10426 ] Correct probabilities for NPE checks in native - image .
Fix Snapshot Corruption in Edge Case ( )
Target container mapping fix ( target has higher priority than target name ) . Task log flush fix
NEw version dialog redesign ( release notes field )
[ grid ] : Limit visibility of NodeServer logger
[ SECURITY - 76 ] [ SECURITY - 88 ] Adjust test to policy changes .
Remove getter for SchemaMetaData ( )
[ FLINK - 19218 ] [ core ] Remove inconsistent / misleading locality information from Local File Splits .
Reduce memory usage in MarkBasePointersPhase on JDK11 .
Pre - allocate JNI C structs in the image heap
[ GR - 22562 ] Return method reference instead of the anonymous class .
bugfix : correct LockManagerTest . concurrentUseAbilityTest assertion condition ( )
faster handling of a case where each doc has multi valued values fields in histogram facet
Refine the DegradeRule and add validating logic for the two new attributes
PostgreSQL CLOBs
[ hotfix ] [ tests ] Use self gateway in TaskExecutorPartitionLifecycleTest . testJobMasterConnectionTerminationAfterExternalReleaseOrPromotion
[ TEST ] enable inline scripts on demand in bw comp tests
Polishing
parentFile could here still be null in case of Maven 2 . x
Preserve repositories ' rooted paths
[ inline - info ] step 1 of de - aliasing MethodInvocation assumptions
improved kingbase8 support , fix
adapt to upstream changes
MINOR : Fix deadlock between StreamThread and KafkaStreams
Track changes [ ci skip ]
Replace AtomicReference < Boolean > with AtomicBoolean in AbstractServerHttpResponse . writeWith ( )
Use DataBuffer . write in CharSequenceEncoder
Fix Some Bugs
Redesign MockMvcWebClientBuilder API
Attribute selector dialog fix ( selection order number )
use isJavaLangObject ( ) in BasicArrayCopyNode
Data editor refresh - respect filter values
Add callback for publication of new cluster state
KAFKA - 3691 ; Confusing logging during metadata update timeout
Simpler FilePath constructor .
set the node service on the discovery to get the additional service level attributes ( issue )
core : fix signature processing for local variables
NPE fix
bug fixed wall filter & licence info
Upgrade to Spring Security 4
added HotSpotObjectConstant . getSuperclass ( )
modify assertEncryptRule ( )
Add configurable property for JWK encryption algorithm
[ hotfix ] [ FLINK - 19183 ] [ hive ] Fix compile error by TableEnvHiveConnectorITCase . testTransactionalTable
DouniaBerrada : On behalf of JuanTrujillo ( with some modifications ) , fixing typos in flick tests .
[ TEST ] Don ' t use shaded API
[ GR - 7322 ] Do not count transferToInterpreter into invalidations .
When Watcher starts verify that all index templates have been added , otherwise add the index templates
Write extractor dumps to testdata
Introduce @ RestrictHeapAccess . NO_HEAP_ACCESS
Extract WebMvcProperties
Ignore unknown fields if overriding node metadata ( )
[ GR - 23863 ] Be careful with MultiGuardNodes in GraphUtil . killCFGInner .
Refactor how PyCommon manages its transitive sources
[ hotfix ] [ test ] Fix StreamConfig propagation to StreamTask in StreamTaskMailboxTestHarnessBuilder
fix typo of Bootstrap in sharding - proxy ( )
Use RegEx for test to make Windows happy
GT - 3414 revert Iterable change .
sulong / toolchain : avoid " argument unused during compilation : ' - stdlib = libc + + ' " warning [ GR - 21095 ]
Tests web app properties access properly
Support profiler command on arm64 . ( )
PEGraphDecoder should respect InlineControl
Scheduling : remove dead code in addToLatestSorting
[ ] Fix calculation of writerIndex ( ) which was incorrect and so result in truncated DatagramPackets when using NIO
Avoid deadlocks in cache ( )
Allow custom FrescoController implementations to be passed to FrescoContext
Make native linker for LLVM compilation customisable
Polishing log message and comment
Fixed signature compatibility issue .
Add rest of indices primaries stats .
[ FLINK - 17790 ] [ kafka ] Fix JDK 11 compile error
improved reference support . for issue
Polish
Change additional groups copy to match iOS .
refactor QueryHeaderEncryptorMetaData
Fix SDK int check in TextInputLayout
index bug fix within gc
[ FIXED JENKINS - 15439 ] Jenkins build records lazy - loading failed to load some of my jobs
Use class names instead of fields to detect Java version
[ GR - 5528 ] Do not hold a static reference to engine accessor .
Added uniqueness hints to k - means iteration example .
[ hotfix ] [ runtime ] Propagate execution graph execution state update result to JobMaster
repair error in plan when order is include - like PLAN ( RDB ORDER RDB )
Fix playback of badly clipped MP3 streams
sulong : fix ll level debugging for bitcode files without debug information
We ' re on Java 8 already .
for , adjust root hook start
Remove workaround . The ASM bug is fixed . And we use ClassRemapper now .
Raise error if we find an unknown type in native . rule ( ) .
[ GR - 2872 ] Improve handling of PiNodes in address expressions .
Handle multiply - defined attrs in AndroidResourceClassWriter
[ java ] We should not call bazel when we are not in dev mode
DDLParserUtils rename to VisitorUtils
fixed NPE
auto input params and print log
Remove USE_CHORD_PITCH
Add initial Slider implementation
configure auto expand for dataframe indexes ( )
Oracle 8 support ( fix joins in metadata queries )
use temporary proxy for locations in snippets
No obvious reason why the specialized doDoDelete in AbstractProject could not be pulled up into AbstractItem .
Truffle - DSL : made type name verification more robust . Added some tests . ( GRAAL - 468 # resolve )
LocalSpawnRunner : on Windows , don ' t use the process wrapper for now
[ GR - 26725 ] Increment call count for call nodes in first tier compilations .
Update to elasticsearch 1 . 0 . 0 . RC1
fix typo of field ( )
synchronized deadlock fix
Ups Ashley version to 1 . 7 . 0 in setup app
Move regex error to node ( )
sulong : fix debug test suites paths
Refactoring of the OptionsParser implementation to allow overriding the value of an option with null .
Close DrawerLayout on back button press
Mock Data menu item is added to the resultset context menu
[ GR - 11624 ] Don ' t try to instantiate abstract classes .
add ES 6 . 5 . 1 ( on master ) ( )
bazel docgen : also search net / starlark / java for classes
A minor refactoring .
add testcase & bug fixed
Non - transactional keywords fix ( for NoSQL )
fixed issue , support BETWEEN in update statement . ( )
[ FLINK - 19539 ] [ jmx ] Synchronize accesses to port
core : fix class file loading ( )
Prevent inline tracing by default in snippets and stubs .
[ ML ] Increase log level for forecast disk storage problems ( )
Sulong : dependencies are now either a source or the call target if the library has already been parsed .
reuse version key in an actual operation
Added ProgramBuilder support for default languages
Results tab name persistence
[ GR - 3406 ] Faster constant folding during graph decoding .
[ ML ] [ Data Frame ] Fixing default delay set in timesync ( )
[ ] Fix bug in DefaultChannelGroup . close ( ) which was blocking the close ( ) call on ServerChannel
Prettier output .
core : fix warning from dx library
make it runnable using mock data
Move some tests to AbstractEndpointHandlerMapping
Deprecate ReferenceLibrary , use InteropLibrary . isIdentical instead .
improved JSONObject . toJavaObject performance
Makes AWS Elasticsearch errors a bit more descriptive
Removing unused files with the new Types class .
modify cases
Entity figure model refactoring
[ TEST ] fixed error message in REST tests runner
SQL formatter : single line comments format fix
[ hotfix ] [ metrics ] Prevent registration exceptions
Make Converter . andThen ( ) final .
Added_option_to_open_selected_part fix2
Props panel attach / detach logic fix
[ ML ] Remove unnecessary call to get datafeed stats in preview ( )
Bump to 1 . 3 . 3
[ hotfix ] [ runtime ] Make BufferStorage independent with InputGate # getPageSize
chore ( release ) : prepare to release rocketmq 4 . 8 . 0
[ FLINK - 16802 ] [ hive ] Set schema info in JobConf for Hive readers
Add merge boolean property to ExplodeLoop annotation to indicate that backward branches should be merged if possible .
concurrency handling improvemnt
Fixes rss url ; otherwise it gets 302 status code and shows a blank list .
Added feature for min / max width limits on yAxis
[ Monitoring ] Removing unused version . * fields ( )
KAFKA - 5567 : Connect sink worker should commit offsets of original topic partitions
MINOR : Use method references instead of anonymus classes in Errors ( )
[ hotfix ] [ tests ] Fix incomplete buffer reading in S3 test
Simplify CLI integration auto - config test by removing use of Artemis
[ GR - 14565 ] Indicate libgraal in Truffle . getRuntime ( ) . getName ( ) .
, fix test case
Restore identifying org . hsqldb . jdbcDriver as embedded
Removed LocalDateToDateMidnightConverter pre - release since DateMidnight is deprecated as of Joda - Time 2 . 3 ; marked DateTimeToDateMidnightConverter as deprecated
DB2 : Added packages properties
remove unnecessary caching
Finish remove checksum function
[ hotfix ] [ streaming ] Allow to override methods from TwoPhaseCommitSinkFunction
Fix dynamic mapping update generation . ( )
SRC - 89 , for Win2k support using Reg v1 . Only specify the key type when adding the key .
Add a thread timeout method to GlideExecutor ' s builders .
pass thread register into write barrier snippets instead of getting it from the host provider
for 954 , fix use db bug
[ ML ] Don ' t get stats for jobs that are being deleted ( elastic / x - pack - elasticsearch )
Do not eagerly resolve types in lookupType .
Avoid call to MetaUtil . format in GraphBuilder if it is not necessary
SQL script generation fix ( do not add delimiter to the last statement )
Alerting : Add support for history and fix issues around parsing lastRun .
unified ' run ' and ' remote ' in pact - client
TexturePacker , allow user to set the amount of bleed iterations . ( )
SQL parameters in DDL queries config .
Fix crashes on interrupt due to concurrent sandboxfs requests .
Do not warn if failed to mark a void promise as success
Include real type in type casting error message .
MINOR : add ImplicitLinkedHashCollection # moveToEnd ( )
Make InstrumentationHandler # createBindings count with new executionBindings being added by execution event factory .
Additional secure DummySurface device exclusions
Fix inverted condition in Debug . create ( Metric | Timer )
Parser : Allow ` not not x `
Also pass firefox profile to options on start up
Test the Gradle Plugin against Gradle 6 . 1
Use an enum for specifying play modes for animations .
Remove implementation from deprecated truffle_address_to_function builtin .
Fix Maven tool configuration test
, fix PostgreSQLQueryResponsePackets
[ hotfix ] [ filesystems ] Also mirroring s3 . path - style - access .
MINOR : add logfilename to error message when file missing
Correctly copy existing elements when CodecOutputList . add ( index , element ) is called . ( )
Fixed issue : NETTY - 373 Deadlock in ChunkedWriteHandler on channel
apply same fix used on DecryptingPartInputStream to AttachmentCipherInputStream / / FREEBIE
[ GR - 19868 ] Remove graal_detach_threads and poly_detach_threads .
Use FstCodec as default
Add volatile for the stop flag of watcher in ConsulDataSource ( )
Restore use of SpmcArrayQueue in RxRingBuffer
modify test cases
Clear JAVA_OPTS in CLI tests ; avoid permsize config error with Java 8
Convert Value fields to Address .
change default of GraalArithmeticStubs to false
refactor
Assert that detached plugins and their dependencies load successfully
Move dependency declaration out of finally block .
polishing
Automated g4 rollback of commit ed3327827d654d42337bddb053cbd47ed921fe86 .
Improve grouping of Truffle dumps .
Fix InterruptedException when KafkaCollectorWorker stops ( )
Tests - fixed bug in GTable auto - lookup feature
DefaultDnsRecordDecoder compression and index decode bug
[ FLINK - 11897 ] [ tests ] Wait until all tasks are submitted
Use a dedicated id to serialize EsExceptions instead of it ' s class name .
[ GR - 9665 ] Fix a failure of FeebleReferenceListTests with heap verification enabled .
Fix Java 9 compilation issue
Fix DieWithDignity logs assertion ( )
Pass execution info to xml generating spawn . Fixes
[ grid ] Code formatting and improving method name
[ hotfix ] [ network ] Fix wrong string formatting for BufferOrEvent
Use System . currentTimeMillis ( ) instead of System . nanoTime ( ) to protect
JimEvans : Correcting JavaScript to run in IE6
AArch64 : fix branch target overflows and return addresses .
Code review responses
Calling resume in an AsyncFile endHandler will cause a new call to the endHandler - fixes
[ runtime ] Fix for task canceling the order of ( 1 ) resource deregistration ( 2 ) completeness notification
for , refactor PredicateExtractor . getLeftParenIndex
Revise duplicate filtering in SpringFactoriesLoader
[ ML ] Preview of datafeed with aggs should respect chunking config ( elastic / x - pack - elasticsearch )
Refactor ContactPhotoFactory to handle changing local contact URIs .
Make building dll less a hack on Windows
Add did you mean to strict REST params
Initialize sample streams in FakeAdaptiveMediaPeriod .
Introduces - - apple_crosstool_transition , which turns controls the activation of
add Preconditions . checkState
Fixed issue 1034 , ScrollPane crash if no knob drawable .
Fixed - an error shouldn ' t appear if Redisson successfully got the information at least from one sentinel / cluster server .
modify the transaction alert
[ FIXED JENKINS - 15094 ] - Handle null parameter values in BuildCommand ( CLI ) and Parameter classes ( CLI ) .
Increase TruffleGraphMaxNodes to 20000 .
HttpObjectDecoder . invalidChunk ( ) should return LastHttpChunk
[ ML ] Get resources action should be lenient when sort field is unmapped ( )
remote : fix unexpected IO error ( not a directory )
allow = = when one of the variables is the receiver in equals ( )
Sample : Used DIO . showImageOnFail ( . . . )
Analysis : Fix no response from Analyze API without specified index
Fixed changes made in : 36a62a8cc861a11cb4802ee0cd283da88ae1bf90
Fixed unit tests and equals method for ResultSetLiveTest
Refine return type of STRUCT factory methods
[ hotfix ] Fix package private and comments
more options on stress test
[ TEST ] fix typo in get - watch documentation ( bis )
Address flakiness in TomcatServletWebServerFactoryTests
Tests for apple_cc_toolchain .
[ streaming ] Fixed InternalStateTest to avoid printouts + minor StreamVertexTest fix
NETTY - 401 - NullPointerException when ReadTimeoutHandler is
Hide javax . servlet . SessionTrackingMode
DanielWagnerHall : mkdirs rather than mkdir
TexturePacker2 , added flattenPaths setting .
[ fixed ] issue 454 , thanks Thotep .
Canonicalize coversions that don ' t change the type .
[ FLINK - 1951 ] Fix NullPointerException in delta iteration due to missing temp
Work around broken AAC decoders on Galaxy S6
Use a clearer name in DefaultTruffleSplittingStrategy
[ TEST ] Beef up MoreLikeThisActionTests # testCompareMoreLikeThisDSLWithAPI
[ GR - 10903 ] [ AOT ] we should check that VM uses the same GC as one used for AOT library generation .
Fix jvm options expansion on windows ( )
removed compilation error now that some variables are not accessible .
Fixes
Provide MXBeanLookup state cleanup that does not affect image build
[ FLINK - 11183 ] [ metrics ] Move memory metrics creation into separate method
Apparent fix for issue 383 / Moment + Android 2 . 1 issue
Class ExecutionGroupEdge is now thread - safe
JariBakken : Update SingleTestSuite for .
Add experiment shim to toggle checking for cached image data in the image state
Fix issue where scrolling could get stuck .
Text utils refactoring
[ FLINK - 2638 ] [ core ] Add @ SafeVarargs to the ExecutionEnvironment ' s " fromElements ( . . . ) " method .
Don ' t use WavHeader . averageBytesPerSecond
Fix unnecessary boxing and incorrect Serializable
Adapt BWC conditions after backport of ( )
Update the sub class SkipBackendHandler of TextProtocolBackendHandler ( )
Polish
Introduce - - override_workspace_root blaze flag to hand - set workspace_root and mainGroup in xcodeproj .
Add options to enable / disable bitmap reusing
Polish contribution
fix a couple of npe caused by user tapping on a chart which has no data
minor code change to the pow intrinsic to correct a corner case
[ fix ] Truffle compilation canceled itself in case of recursion
typo ( )
Fix JSON parsing of unicode escapes
for , refactor PostgreSQLComParseExecutor . databaseType type from DatabaseType to DbType
Add material . attribute
[ hotfix ] Use own SumReducer in WindowOperatorMigrationTest
Tuned the Code 128 parameters :
Delay LoadHubNode lowering .
[ fix ] GraalCompilerTest : retry compilation in case of a non - permanent bailout
Fix % projectName % for arm .
On WSL , relax inode / device check for image file .
no need for flush / optimize
modify authentication
HotSpotResolvedJavaField : be more precise about fields that are not embeddable ( GRAAL - 290 )
Wrap VerifyError in ScriptException ( )
[ GR - 14242 ] Getcwd syscall for sulong .
Transport : A failure to handle a response might cause the transport to stop working , closes .
Add tests for ByteSequence and fix problems .
Truffle : Update for the CompilationPolicy
[ TEST ] Fix testLimitConcurrentShardRequests failure
Reuse helpers in LayoutModel
[ RECOVERY ] Allow cancel waiting for mapping changes
[ FLINK - 1201 ] [ gelly ] fix ClassCastException and Type errors in mapVertices ; fixes and
Fix syntax . See
Duplicate subexpressions in OneSidedPlatform . java
Fix value assertions cannot assume writ ability ; Reduce recursion depth to one to speed up assertions .
Don ' t show not important warnings
Fix naming to reflect that CEA - 708 is supported too
[ FLINK - 1201 ] [ gelly ] Removed unnecessary edgeKeyType field
Only try to load conscrypt class in tests when supported .
Create directory for Standalone Skylark
MoreLikeThis : Rename fields ( rest / http parameter ) to mltFields , closes .
Add object copying code to AvroSerializer
update to jdk11 jvmci - 19 . 3 - b04
RSV : filter dialog settings apply
improved output of method name under - G : + PrintProfilingInformation
Replace explodeDirectory with visitTree for an on - line visitation of tree artifact contents .
SQL : Fix version loading in JDBC ( elastic / x - pack - elasticsearch )
HTTP / 2 DefaultHttp2ConnectionEncoder data frame size incorrect if error
Call setHandshakeComplete ( ) before the handler replacement ( )
Never lower direct invokes to unreachable
Adds ability to set opacity for Color object ( on behalf of Mark Collin ) .
Polish , remove call of deprecated method
Do not show - - native for Bash launchers and clarify standalone conditions
[ ] Allow using a bounded ByteBuf as the first inbound buffer
Fix DataSpec toString formatting .
KAFKA - 8449 : Restart tasks on reconfiguration under incremental cooperative rebalancing ( )
Add a flag to disable cpu transformer for CppConfiguration
Raise a exception when keyProperty is not found
Improved case insensivity of componentIDs .
Fix tests for modern JVMs .
New methods in StorageUtils
Not use RC4 for renegation as it is not supported in more recent java versions .
BAEL - 3298 Updating names of testcases
Transform DB2 tools into tasks
add units tests
Add file upload test to Spring MVC Test
Use the the artifact category in LinkerInput to determine if it ' s a dynamic library .
RSV filters fix ( do not quote attr name if it is an expression )
External data location configuration fails ( path . data setting ) , closes .
Polishing
[ GR - 6456 ] Print the output path when ShowDumpFiles option is given .
TruffleCompilerThreads does not need to be a stable option , it is only accesed once during startup
DouniaBerrada : Removing clearTextEntry because it ' s not in the Android SDK anymore .
Make sure the test also work on ipv6 interfaces
for Add UseStatementBackendHandlerTest unit case .
Allow / health to be accessed anonymously irresepctive of its sensitivity
Load filter state in separate job
Fix incorrect channel configuration for 8 ch pre - M
refactoring
Render the focused Slider value indicator on top
modify MySQLDataSourceMetaData
test : also wait for paused state
Rename DeferredMediaPeriod to MaskingMediaPeriod .
Change " worker - protocol " execution requirement to " requires - worker - protocol " .
Cat tasks output should respect time display settings ( )
add slfj logger
Added ConvertToClassAction help documentation and tests
FileName is optional in StacktraceElement
Support date math for ` origin ` decay function parsing
Work around broken raw audio decoder on Oppo R9
Fotmat LeafSegmentKeyGenerator .
add testcase for JSONPath
Fix ArrayIndexOutOfBoundsException in DashMediaPeriod
Schema selector fix ( use active editor ' s context )
[ ML ] Give kill a chance to stop autodetect before closing input ( elastic / x - pack - elasticsearch )
[ FLINK - 14926 ] [ state - backend - rocksdb ] ( follow - up ) Make RocksDBResourceContainer immutable
Add test for basic exception .
Do not reply to pings from another cluster
@ Nonnull
Extract property sources from composite when binding
BEP : report more test - result files and add a constants class
Fix compilation .
removed public access to Unsafe in GraalTruffle runtime
Add ignore_above for message . raw field in notifications index mappings ( )
Throw an exception if a config file that exists cannot be parsed
Reduce memory usage of SslHandler when OpenSslEngine is in use
Polish
Mock Data . Save and restore the selected attribute .
change to 0xff
[ JENKINS - 38187 ] Make slaveAgentPort readonly in case the enforce flag is set
KAFKA - 8324 : Add close ( ) method to RocksDBConfigSetter ( )
revised DispatcherServlet ' s last - modified handling to properly work with scoped controllers
RSV associations fix
Fix NPE bug with fail / reply on send with no reply handler or publish
1 . 1 . 21
, DatabaseType . MySQL = > DatabaseType . PostgreSQL
Fix test with dates
[ FLINK - 2578 ] [ tests ] Fix stability of TaskAsyncCallTest
Print environment for all AbstractAction in aquery text output .
modify the monitor api
Widen visibility for SkyframeExecutor . setSkylarkSemantics
Tests : fix RangeQueryBuilderTest , only use timestamp when date mapper is available
Fix broken build
Minor improvements in comparator tests
Use LWJGL Sync class instead of Thread . sleep for JGLFW .
Bugfix in inlining
Add directory output information to target completion events in BEP .
Update the DefaultExtractorInput ' s peek buffer length on each write
Add missing onLoadStarted callback to Extractor and SingleSample media period .
Make Splitter . on ( String ) redirect to Splitter . on ( char ) if the String has one character .
Finnished porting TPCHQUery3 and replaced the deprecated class by the new one - - untested
formatting fix
Made libgdx version configurable in gdx - setup
[ FLINK - 20493 ] Increase timeouts for building FlinkContainer images
[ GR - 20030 ] Extend Truffle privileged code detector to detect access to Unsafe embedded in the boot image .
SL : added exemplary uses of new profiling utility BranchProfile to SL .
Fix wrong position of linear legend form
for , Only DQL and DML Parser relative test cases need execute Placeholder 5th .
[ FLINK - 4251 ] [ Rabbit MQ ] Allow users to override queue setup in order to customize queue config
Changes in test file
Dump debug counters to System . out via shutdown hook if option is enabled
Allow PORTFILE to always override the file to use
Remove the last readFrom from xpack
Support warn / default strict deps in java_common . compile .
[ JENKINS - 46523 ] - Make UnixReflection class compatible with Java 11 , remove pre - Java 8 reflection ( )
Add a new " resource_name_obfuscation " feature string .
function_score : use query and filter together
[ FLINK - 12004 ] [ core ] Make Pojo types accept chainable setters
Refactor ShardingSphereAlgorithmBeanRegistry
Remove two max size 500 limit . ( )
Add UriTemplateServletHandlerMethodTests
Support " regexpi : " case - insensitive regexps .
[ jobmanager ] Fix potential null pointer exception in jobmanager webfrontend
made helper methods private
Renaming to SINGLETON
Fix test to be non racy anymore and remove irrelevant tests
fix failing IsolatedPluginTests ( change numNodes to 0 )
fix WatchStoreTests
Fix serialization bug in painless execute api request ( )
Polishing
Test Cases - XStream Object to - from json
[ GR - 10251 ] Improved help consistency .
KevinMenard : Log if the Matcher thread manages to die somehow .
DB2 : In cas of special register for current schema return null , use the connection userid as current schema
Improve stability of shardsCleanup test
Querying mapping on a non - master throws an error , closes .
TraceRA : replace TraceRAbiDirectionalTraceBuilder and TraceRAsingleBlock option with TraceBuilding
Explicitly build terminal with jansi .
Enable customization of RestTemplate that retrieves JwtAccessTokenConverter ' s key
Add AV1 HDR profile recognition
[ FLINK - 20540 ] [ jdbc ] Fix JdbcCatalog connection exception when baseUrl doesn ' t end with slash
Prettier output
Deleting a non - existent warmer shouldn ' t cause request to hang
sulong : rethrow the UnsatisfiedLinkError in NFIContextExtension # loadLibrary [ GR - 20440 ]
Add variables parse
[ JENKINS - 30084 ] some stylish changes
[ ML ] Fix bug initialising ML MetaData ( elastic / x - pack - elasticsearch )
[ GR - 7790 ] Treat JavaThread : : _threadObj as immutable .
better error message on nested sorting
Fail invalid incremental cluster state writes ( )
2287 - ImageClassLoader doesn ' t need to hold on to Methods and Fields which do not have annotations
Navigator : new object open fix
Shorten the player type
HostedField . getOffset returns HotSpot field offset
Move XContent rendering of ClusterIndexHealth to the class itself
Fixed ' globalJobListeners ' / ' globalTriggerListeners ' to work with Quartz 2 . 0 & 2 . 1 as well
Sulong : command line libraries can be native libraries
Use " date sent " as timestamp for push , " date received " for SMS .
[ FLINK - 14397 ] [ hive ] Failed to run Hive UDTF with array arguments
work around the tricky bug about machine config page
Fix possible NullPointerException in HotSpotMethodResolvedImpl . toJava
JDKInstallerTest compiling against HtmlUnit 2 . 17
Sulong : adding truffle boundary for blacklisted svm functions .
up to java 1 . 7
Polish " Permit use of https for configuring Prometheus push gateway "
SchemaStatVisitor support cast expr
fix typo
[ FIXED JENKINS - 37561 , CID - 1205051 ] - Prevent resource leak in AntClassLoader # findClassInComponents ( ) ( )
Fixed mixed keys for CoGroup , adapted test cases , code clean up
for fix bug that DAL didn ' t merge
core : fix catch block argument if move - exception instruction is missing ( )
SimonStewart : The headless build server is having a hard time with UTF - 32 . Switching to UTF - 16 instead
Deflake MemoizingEvaluatorTest # shutDownBuildOnCachedError_Verified from the deflaking I just did .
Simplify code
[ FLINK - 12436 ] [ hive ] Differentiate Flink database properties from Hive database properties
hack for log
[ GR - 2741 ] Simplify RegisterAllocationTimeBenchmark .
wait for mappings to be created on other nodes before returning from the put mapping request
for checkstyle
Moved CombinedIterator to org . jboss . netty . channel . group because it ' s only used there
Fix min api check in RobolectricUtil
Fix test bug : norms are on by default on _all .
ProxyHelper handles proxy URLs with trailing slash .
Add more documentation for rule . outputs
[ FLINK - 10567 ] [ state ] Fix TtlStateSerializer doe not propagate all field serializer during duplicate ( )
Issue : Close DiskLruCache . Snapshot after usage
Cleanup after unexpected exception from external delegation call
AArch64 : Better instruction sequence for constant move
SQL Server : tables / procedures search fixed
Fix ToolchainResolver to show an error when a toolchain type resolves to an
allow to nullify routing values when providing then through REST API
Add runtime_deps to android studio proto .
[ GR - 7679 ] Graal does not respect ObjectAlignmentInBytes VM option
Report what RemoteSpawnCache is doing .
Issue 1789 PDF417 encoding fix for 1 - character latch into byte mode
Avoid stateful MethodParameter nesting level changes in MVC processing
Upgrade to Ehcache 3 . 1 . 1
disable future timeout tests since it breaks travis CI ( )
[ hotfix ] [ qs ] Combine logging statements in QueryableStateUtils # createKvStateServer
rename PropertyNamingStrategy , add ' case ' postfix .
KAFKA - 8869 : Remove task configs for deleted connectors from config snapshot ( )
Polish contribution
Close InputStream when receiving cluster state in PublishClusterStateAction ( )
- Test case for expiration renewal not working after connection problem ( refactor ) ;
Do not expand depset twice and store expanded in ctx . actions . run inputs .
Matchers is now deprecated in Mockito 2 , it ' s now replaced by ArgumentMatchers
Added foreach for array types .
Changing pmovzxbw and shlxl into using predefined emit sequences .
Include stack trace with ILM error in explain output ( )
Fix membership diffs that occur during a GV1 - > GV2 migration .
[ FLINK - 18055 ] [ sql - client ] Fix catalog / database does not exist in sql client
Simplify duplicate vertices check .
rewrite SslHandler ' s flush0 message to match docs
Fix corner case when short - cutting gotos in the bytecode parser .
Keep - alive configuration
[ GR - 3557 ] Fix regression in reading options from a file .
[ java ] Changing tests to follow atom changes
Add debug log for retention leases ( )
KAFKA - 8559 : Allocate ArrayList with correct size in PartitionStates ( )
DanielWagnerHall : Search for firefox before version - specific firefoxes . Fixes issue 2629 .
the ttl purger might have already deleted the documents .
MySQL / MariaDB tables read fix ( quote identifiers )
Log additional information about the object in question if things are failing .
Add version info in introspection data .
release searcher in finally block rather than relying on an exception that is thrown
Add an TrackSelector base interface
Grid fix ( column pack lead to data reread )
Fix off - by one error in TransportSearchActionTests
Perform Eager Parsing before Calling Structors
Automated g4 rollback of changelist 268731150
Use TestPropertySourceUtils to convert properties
Use same logic for DASH manifest reloading for all cases when manifest is invalid .
these two fields never get serialized , due to the writeReplace method .
Javac workaround .
Test that close of Context with bound Engine closes Inspector ( GR - 25203 ) .
Multidimensional arrays
SimonStewart : Attempting to fix the build
[ streaming ] solve issue
Fallback to TypeUtils . isAssignable if the message class hasn ' t been compiled yet and cannot be found
Update calls to isHostConfiguration to correctly use isToolConfiguration .
1 . 2 . 35 - SNAPSHOT
Fixed issue in which DashChunkSource didn ' t make use of the most recent
Truffle : do not throw permament compiler exceptions into guest languages .
bug fixed for wall - filter . for issue
Minimal fix for non - default cenc name prefixes in DASH manifests
Accidental Java 7 dependency in test .
KAFKA - 9032 : Bypass serdes for tombstones ( )
[ TEST ] declare support for groovy - scripting feature in the Java REST tests runner
Collapse ConstantsWalker into RuntimeMethodInfo .
Error in Query for cache
[ GR - 13311 ] TruffleFile doesn ' t work in the root directory .
Removed ZoneId code
Close streams on exception
[ streaming ] Pojo wordcount example added to streaming
Changelog and cleanup for [ JENKINS - 8334 ]
Secure storage reset fix
Minor bugfix concerning addXValue ( ) .
[ TRANSLOG ] Fail # snapshot if translog is closed
Update DB version to ensure older clients update ( see Issue 254 )
use after class
Changed how recurring callbacks are paused around monitorenter / monitorexit .
Enforce limit on classes loaded by Spel compiled expression loader
fix some typo regarding token macro in MAVEN_OPTS in maven plugin
[ FLINK - 1201 ] [ gelly ] fixed type info for the outEdgesFunction
for , make consist with ShardingInsertOptimizeEngine & EncryptInsertOptimizeEngine ' s EncryptInsertOptimizeEngine ( ) & fillPlainInsertValue ( )
Specialize ( baseline ) emitPrologue for AMD64 and PTX .
Update to Truth 1 . 0
Truffle : detect inlined recursive calls and introduce depth limit
Jackson2ObjectMapperBuilder prefers Jackson 2 . 6 JavaTimeModule over JSR310Module
inet , uuid and other other column types values quoted in generate sql insert
add request overflow check to OnSubscribeRange
[ SPARC ] Fixing structure of fpops to avoid javac / findbugs complaints
MeshBuilder error using Short . MAX_VALUE vertices ( )
[ GR - 17256 ] Add missing specializations .
Fixed java_stub_template . txt after disabling runfiles on Windows
Add worker support to single jar .
Reset connection before delegating to handler
[ FIXED JENKINS - 11592 ] NPE in subversion polling of Maven jobs
SQL editor fix : disable filters for multi - results queries
Test for cyclic type declaration handling in TypeDescriptor
[ GR - 11513 ] Coarsened synchronization in CompilationWrapper to avoid transient test failures .
Only lowerToIf , when stage does not allow FloatingGuards
Added 1 . 3 . 9 - SNAPSHOT and 1 . 4 . 4 - SNAPSHOT versions
Rename interop . java test package .
Cache splitting options in direct call node
[ GR - 26569 ] Fix for non - existing native library files on OSX Big Sur .
SecureRandom replaced to ThreadLocalRandom in RandomLoadBalancer
Enable AMD64Move . PointerCompressionOp to load constants and permit subclassing of CompressPointerOp and UncompressPointerOp
Make Config . getClass return a meaningful type
[ added ] SplineTest
Set timeout for running process
inherit from the parent .
Add exception type descriptor to TCK .
Fixes . Broken log4j 2 . x integration with rc2 .
Added a test that improves test coverage of the nested class MapViewOfValuesAsSingletonSets ( via asMultimap ) .
Fix playback of postrolls with multiple ads
Do ptrtoint conversion more lazily for foreign objects .
Update mock injection example to make the behavior clear
Prevent reply and mark as read abilities while Signal is locked .
Add support for stable array constants
Work around printCFG / AbstractBlock issue .
temporarily disable encrypted backup
Fix a NPE in a testcase when running via ubuntu
Call incrementalPrepare directly from prepare
[ SPR - 8387 ] Introduced hasResources ( ) in ContextConfigurationAttributes ; plus minor polishing .
Clarify docs for required providers and legacy migration
Fix error message for - - jvm if it is not available
Grants in DDL
when asking for _source with script fields , the _source is not returned .
[ FLINK - 3535 ] [ runtime - web ] Decrease log verbosity of StackTraceSampleCoordinator
Add leaf snowflake key generator and related functions .
Fixed intermittent failure on AsyncMultiMapTest # testMapRemove
BAEL - 1572 : remove quotes
DF : replaced safeGetString ( ) bysafeGetStringTrimmed ( ) when getting the key column
fixed some compiler warnings
Next development version
Primary replica resync should not send ops without seqno ( )
Fix SpEL JavaBean compliance for setters
Remove use of DX in D8 CompatDx tests
BAEL - 2899 : separating static and dynamic classes examples
Display the required version back .
Add case insensitive prefix and wildcard to ' version ' field ( )
Introducing a way to change the name , shortName and path attributes of a source via the withXYZ pattern .
Thread profile through to capabilities in firefoxdriver constructors
Respect canBeInlined in Truffle partial evaluation .
Migrate InlineAssemblyTests to new test infrastructure
fix similarity tests
simplify usage of IncrementalCanonicalizerPhase , use incremental canonicalizer for GuardLowering
Install correct graph builder plugins for FastPE .
fix : PROPFIND request should not redirect to GET request
* Integrated Feedback from
Rework WebSocket server handshake to move the handshake logic to ServerWebSocketImpl
LoadTimeWeaverAware beans are consistently being created early for JPA weaving to work reliably
Fix comment and rename blob_container to blobContainer
Do not throw an exception when failed to get a header
Consolidate HudsonPrivateSecurityRealmTest & HudsonPrivateSecurityRealm2Test .
change ObjectMap to ObjectIntMap
Don ' t show player controls unless key press is handled
RList refactored
[ refactor ] TraceRA : remove SSIBuilderBase
Support sibling columns , with _cat / indices ? pri as first go .
[ GR - 9420 ] Update truffle import .
Filter menu fix
Added control over Query used by MatchQuery with there are zero terms after analysis
add unit test case for merger 5th version
[ RIP - 16 ] fix test case error
Remove requesting members if they are directly added to the group .
Support Bitfields of char Type
[ FLINK - 19225 ] [ connectors ] Various small improvements to SourceReaderBase
Replace CyclicBarrier with CountDownLatch
Scaffolding for overriding DirectoryListingStateFunction .
For , implement getVersionColumns
Fix crash for 4 . 1 devices
Fix bug in explain for function_score queries .
Use Eager Source Locations for * . ll Instructions
Remove deprecated constructor from ` FirefoxBinary `
[ FLINK - 19448 ] [ connector base ] Explicitly check for un - expected condition that would leave an inconsistent state
polish
[ FLINK - 10234 ] [ runtime ] Fix ambiguous lambda usage
Change action construction to use built - in param file support .
Set the platform name in the build event stream to the CPU instead of the toolchain identifier .
Fix ILM status to allow unknown fields ( )
MINOR : change default TX timeout only if EOS is enabled ( )
Fix probabilities for new array instances .
Pass a locale to a getLowerCase call in BlazeRuntime .
Fix issue 3661 ( )
am 2d0bc6c8 : am 86745db1 : am ee460c94 : am 024b3639 : Fix NPE on orientation change
AndroidSound # play / loop now returns - 1 on failure , to match other backends
[ TEST ] Reduce stringency of breaker assertions
fix wrong parsing of useDisMax with number
Added efficient contains ( ) method
sql parser support drop table
Fixed bug in charset decoding logic .
Bug fix : fix 420 , fix negative waitTime in RateLimiterController and WarmUpRateLimiterController
Make serialization of @ ConfigurationProperties beans more defensive
[ ISSUE ] fix failure test cases for acl mqadmin command ;
improve fixing negative query with adding the optimized match_all query
Excpetion = > Exception
Use a safer way of avoiding artifacts
Fixed some collections not respecting Collections . allocateIterators .
Add @ NonNull and @ Nullable annotations
Add ec2 specific network hosts , closes .
Correct settings prefix for the crypto thread pool ( )
Extract closure grabber into AstUtils
Align with breaking API changes in Spring Data MongoDB
Suppress remote task times with async execution
fix expcetion msg comparison
am de947589 : am f6e9a17c : am 7afe341e : am 6a93c2eb : Merge " Add typeface support to TextInputLayout " into lmp - mr1 - ub - dev
minor debug code cleanup
Add some leeway for finding additional tracks in PsExtractor .
Generic driver : supprot stored procedures filter
Make the return type ImmutableList
Fixed some minor SelectBox bugs .
Generic column read workaround ( Clickhouse bug ) .
Add probabilities to NewObjectSnippets . allocateInstanceDynamic
Fixes . Allow linking children to multiple parents in multiple
leave minimum version at 176
use multi - release jars to remove reflection from HotSpotInvocationPlugins
Make BoxNode extensible .
Migrate Truffle * MinInvokeThreshold options to - - engine options
vbo - > VA
Use sum of unscheduled blocks at merge point .
[ ML ] Add notification when no suitable nodes exist to allocate job ( elastic / x - pack - elasticsearch )
[ ] Support write - spinning also in native transport
An improvement to unicast discovery to also ping nodes the node itself received a ping from .
Only associate JTA transaction manager to JMS factory
Replace indexed loop with for - each java5 syntax ( )
SPR - 6467 - Allow ContentNegotiatingViewResolver to be strict ant return a 406 if no view found
Improve error reporting in test
[ htmlrunner ] Allowing relative path to the suite file
Fixed possible NullPointerException when cloud instances are reused
Remove dependency to old LLVM Types from LifetimeAnalysis
catch out of memory exceptions while inspecting download of large files
Fix proxy mappings should not be tested in AOT mode .
[ GR - 12649 ] Mimetype content / unknown and different if started from JAR .
Complete channel initialization in the event loop
BAEL - 1473 ( )
In MethodCallTargetNode . canonicalize , uniqueConcreteType . resolveMethod can return null in some cases
Added isEqual methods to MathUtils
for : refactor DQLIntegrateTest
[ GR - 21031 ] Making T - Trace sources debuggable .
Polishing
[ FLINK - 13774 ] [ table ] FieldComputer should return ResolvedExpression
Per Daniel - - don ' t check phone ' s preview size as we want to use screen size if possible
fixed , optimzier code
[ hotfix ] Harden YARNSessionCapacitySchedulerITCase # testTaskManagerFailure by removing timeout
Reduce the amount of times Task . getFullDisplayName is invoked .
[ FLINK - 18087 ] [ yarn ] Fix uploading user artifact for Yarn job cluster
[ JENKINS - 31162 ] @ jglick ' s comment was addressed . A getEffectiveClazzFor ( ) method was added to determine effective clazz of CustomOrganizationFolderDescriptor
Dangerous SQL confirmation ( warning message )
Connection error message box fix
PG : enum values order fixed
refactor : Use ' of ' instead of Stream ' builder '
fix unit test failure
refactor hint example .
[ JENKINS - 58993 ] we need to set configLoaded earlier
Extended job manager plugin interface to expose possible profiling dependency
disabled jobs shouldn ' t be buildable .
Fix potential issues with ConversationDataSource boundaries .
Fix MockHttpServletResponse HTTP status update
PostgreSQL : trigger comments support
Macro expansion rule now supports $ { ABC } syntax as well as $ ABC .
Add support to VP9 extension for overriding the native library names
[ FLINK - 3390 ] [ runtime , tests ] Restore savepoint path on ExecutionGraph restart
Allow null DuplicationReplacement in addDuplicates .
[ SPR - 8089 ] re - ignoring JMX tests that require the presence of jmxremote_optional . jar .
Fix test failure for watcher with API key in upgraded cluster ( )
[ FLINK - 17697 ] [ jdbc ] Fix JdbcFullTest . testEnrichedClassCastException is failed in JDK11
OpenSslEngine . getSupportedCipherSuites ( ) must return java names as well .
Do not clear all configurator instances when override is empty ( )
Make a rogue test case honor - - experimental_dynamic_configs = notrim_partial like everywhere else .
Add orchestration spring support 14th .
Fix for Observable . toList failing with multiple subscribers
fix ci failed issue
2 . x : fix Completable . concat to use replace ( don ' t dispose old ) ( )
SPDY : SpdyHeaderBlockDecoder should not discard input data
Truffle - DSL : fixed wrong equals / hashCode in Template .
fix auto test error
+ bugfix : " Fatal Signal 11 " on some devices when rotating lwp in preview
[ Fix JENKINS - 44663 ] Make Basic authorization header to be case in - sensitive ( )
[ FLINK - 17578 ] In StreamEdge , use output tag in hashCode and equals
[ TEST ] Properly clear the disruption schemes after test completed .
MINOR : Use EnumMap / EnumSet if possible ( )
improve canonicalization from IfNode to NormalizeCompareNode
Release resource after use in ConfigParserTest ( )
KAFKA - 9851 : Revoking Connect tasks due to connectivity issues should also clear the running assignment ( )
Fix ForwardingDrawable support for null underlying Drawables
Add assertion check to FieldFilter . and ( FieldFilter ) method as well
[ hotfix ] [ network ] Rename RecordWriter # closeBufferConsumer ( ) to closeBufferBuilder ( )
prevent test failures from jython localization bugs .
Move statement
RSV : multiple rows add / duplicate
iOS va_list passes arguments as an array on the stack .
[ GR - 19512 ] Ensure that CodeInfo objects are not freed prematurely .
Test : Expect extra Exception in die_with_dignity ( )
RELNOTES : Fix FDO_STAMP_MACRO to only be set when fdoBuildStamp is not null .
Closes issue : PURE_BARCODE fails on certain dense QR codes
Extract the conditional for checking if we need to resize image into a new method .
Create a Bootstrap for config - related libraries of skylarkbuildapi .
Fix occasional crash on Android 7 direct reply
changed name
upload date in description now matches newpipe ' s language
fix test case for jdbc
modify the cross handle default sort type
Fixed avatar icons not rendering in contact selection view
[ hotfix ] [ checkpointing ] Add VisibleForTesting annotation for related methods
Apply changes recommended by Jetty team to JettyEmbeddedErrorHandler
TraceRA : number instructions during interval building .
Better default size for global index - > alias map
Fix formatting .
Fix test code formatting
Add support to run actions into FakeDataSource
GO delimiters evaluation fix
modify the status
[ ] Prevent IllegalBufferAccessException which could be triggered if inboundBufferUpdated ( ) was called once the channel was closed
[ FLINK - 13192 ] [ hive ] Add tests for different Hive table formats
[ FIX JENKINS - 38615 ] Add user to restart log message
PublishSubject to match BehaviorSubject
Commit id plugin ( )
[ FLINK - 7783 ] Don ' t remove checkpoints in ZooKeeperCompletedCheckpointStore # recover ( )
Add extra test for partial overrides
Indices exists API , closes .
delete invalid function
Do not swallow fail to convert exceptions ( )
Enforce main thread for api calls .
Add missing replacement of object array elements when building heap histogram data
[ FLINK - 8928 ] [ QS ] Improve server binding error message .
Find the entire relocatables range in the mappings list , fix range checks and style .
Add isSource to Android aspect ArtifactLocation .
JPA EntityManagerFactoryUtils silently ignores IllegalArgumentExceptions from setHint calls ( SPR - 7947 )
Workspace password handle fix + editors / views lazy init fixes
Fixed custom hunspell dictionary directory
Only override gapless data if set in ClippingMediaPeriod
[ hotfix ] [ flip6 ] Harden JobMaster # triggerSavepoint
fix : resolve error in SkipMethodArgsAttr if method not yet loaded ( )
Un - @ AwaitsFix ExplainIT
[ Test ] Remove dead code from ExceptionSerializationTests ( )
Clone value arrays in CompositeValue . clone ( ) .
[ ENGINE ] Remove flushNeeded in favor of IW # hasUncommittedChanges ( )
[ TEST ] Fix issue with bulk security rejections
Kafka - 6693 : Added consumer workload to Trogdor ( )
refactor
Consistent throwing of last UnsatisfiedDependencyException if available and no constructor resolved
Add PropertyUtil support spring boot 2 .
TraceRA : minor cleanup in the local LSRA data flow resolution phase
[ JAVA - 621 ] Fixes due to minor article changes
Replace map with set . ( )
Followup for elastic / elasticsearch
1 . 2 . 56
do not assert GeneratedKeyToken
[ streaming ] [ scala ] [ api - breaking ] StreamExecutionEnvironment API update
LoadJavaMirrorWithKlassPhase : ConstantLocationNode . create already adds the node to the graph , remove redundant call to graph . unique .
Sometimes the depth plunges precipitously past 0 . Check for us being lower than expected rather stopping where we want to .
Align with breaking API changes in latest Security snapshots
Allow search to continue when sort field is missing from type mapping , closes .
JENKINS - 33770 - not all paths restricted during SetupWizard
jdk 1 . 5 support
[ ] Fix bug in ReadOnlyByteByteBufferBuf where get operations threw ReadOnlyBufferException
The real reason for this CL is to claim commit : )
sulong : reformat resolveRenamedSymbols C + + support
for : IllegalStateException = > ShardingRuleException for BindingTableRule
improved error diagnostics
[ FIXED HUDSON - 2373 ] in 1 . 316 .
[ LOGGING ] Log an error message when we fail to parse a ` Watch ` or ` WatchRecord `
Improve SystemRuleManager and refactor checking logic of maxCpuUsage strategy ( )
fix : correct arg replace in PHI instruction ( )
Catch all thrown exceptions in CEntryPointSnippets , return proper completion code
By default , we don ' t cache GCE API calls
[ java ] Changing simple assert to a wait
Build vertx instances using the test method so they are cleaned up after tests
2 . 11 . 2 release notes and version bump
Prevent leaking of the Thread . interrupt flag to other LoadTasks
Select row count for custom query
Add a way to override ad media MIME types
add option to draw limit lines on top of data
[ hotfix ] [ runtime ] Small improvements in log messages for Task and RocksDB Backend
fill TableToken
Better diagnosis for null parent .
Inconsistent type info fix ( workaround )
Support aspect ratio fill mode for AspectRatioFrameLayout
Fix URL methods in JarURLConnection
ArrayCopyNode : cleanup , unrolling for fixed length arraycopy
Test that client - id can be hyphen or camel - cased
Float numbers formatting fix ( too high precision )
[ GR - 16689 ] Assure that the same lock object is used for both lock and unlock .
[ FLINK - 18168 ] [ table - runtime - blink ] Fix array reuse for BinaryArrayData in converters
[ hotfix ] Rename TimeReaderFunction to TimerReaderFunction
[ hotfix ] Reuse getMaxJvmHeapMemory ( ) in EnvironmentInformation # getSizeOfFreeHeapMemory ( )
[ ML ] Add datafeed run time fields integration test ( )
improved JSONPath . extract
refactor code style .
DanielWagnerHall : Force ChromeDriver to flush data to sockets before closing them - * massive * stability increase
Remove Timeout in Blocking Iterator
Ensure we only add OpenSslEngine to the OpenSslEngineMap when handshake is started
Add some color to better distinct the objects
Allow multiple PSSH boxes for same system .
Fix content progress reporting if there is no preroll
Revert " [ FLINK - 14971 ] [ checkpointing ] Remove lock of PendingCheckpoint and introduce IO lock for PendingCheckpoint "
[ inlining ] consumer becomes initializer of the probabilities map
[ FLINK - 11164 ] Check for sentinel values when creating new Kinesis ShardIterator
use bounded wildcards for errorHandler ( fixes ) ( )
refactor MasterSlaveRuleConfiguration
[ SECURITY - 120 ] Do not print a warning with stack trace just because we are using a 2 . x servlet container .
scalar constructor fix
adapt to upstream changes : enableMockModules = > getMockPlugins
Fix indentation .
Added page size to atlas format .
Redirect handler should not decode query params
EQL : Improve request logging ( )
modify unit test
Fix conversation jumping when loading at last scroll position .
Use System . nanoTime ( monotonic ) to detect idling delays .
Adjust corners of MaterialShapeDrawable ' s stroke to draw the stroke perfectly over the fill .
Store : Delete index folder if all shards were allocated away from a data only node
Added an illustrative comment in the ParamMap precondition check . Added methods to take a formatted message in Preconditions .
delete rewrite ( ) interface
Allow users of Blaze Lexer to explicitly specify the line - number table of a file .
Always increase timeout for packaging tests ( )
Do not inherit directly from AudioTrack . StreamEventCallback
Make - - compile_one_dependency more " configurable attribute " - friendly :
[ FIXED HUDSON - 6673 ] Add setting so ListViews may show only enabled or disabled jobs .
sulong : make loadC + + Libraries off by default
Apply themed context to SingleRecipientNotificationBuilder
SQL Server schema read fix
with Java5 dependency , there ' s no need to use Ant for environment variables .
SQL : JDBC : fix temporary directory locked test errors in Windows ( )
[ FLINK - 17634 ] [ rest ] Reject multiple registration for the same endpoint
native image : cas vm local add missing state after .
Bump version to 2 . 5 . 0 - beta2
fix : incorrect init values of inherited fields
fix MultipleRenderTargetTest : MRTShader was inconsistent ( )
Clean up DownloadManagerTest
[ java ] Adding tests for Window . fullscreen ( )
add explicit type missing exception when trying to search on a type that is not registered
Fix cycles com . google . common . cache .
avoid code generation for typechecks that are only used for anchoring
dbeaver / dbeaver - ee adding many commandLine parameters fixed
DanielWagnerHall : Fixing firefox warnings
Fix processing of paths from APIOptions
TAB - > WS
getPropertyTypeDescriptor bug fixes
Fix SearchInputIT . testSearchInlineTemplate
bug fixed for BigDecimal toPlainString .
Handle pre - 6 . x time fields ( )
[ FLINK - 6581 ] [ cli ] Correct dynamic property parsing for YARN cli
see 04 / 27 log
[ FLINK - 14439 ] [ runtime ] Enable RestartPipelinedRegionStrategy to leverage JM tracked partition availability
Fix uncached execute generation with varargs evaluated arguments . ( GR - 21203 )
Fix issues with failed cache loads
Changed RecordSerializer interface
[ FLINK - 11768 ] Update TypeSerializerSnapshotMigrationITCase for Flink 1 . 8
MINOR : Bump timeout for sending records ( )
remove log statements and use testComplete and await instead of our own latch
Make AbstractQueueVisitor . getTaskCount public
adjust ShardingExecuteCallback ' s api for SkyWalking plugin
MINOR : Document " high watermark " magic value for delete records request
- Remove unnecessary catch block
[ FLINK - 2376 ] Raise the time limit in testFindConnectableAddress under Windows
[ BAEL - 3646 ] addressing pr issues
Project metadata folder create on demand ( fix ) .
Constraint seek targetGranule within bounds + simplify tests
for SeataATShardingTransactionManagerTest # assertRollbackWithoutBegin
Add logging to AllocationRoutedStepTests ( )
Prevent endless recursion in SezpozModule . resolve .
Refactor unit tests for PostgreWithClauseAppender
[ fixed ] uitexture problem
Fix checkstyle violation in remove plugin tests
Simplify if - condition and remove internal javadoc
More debugging
fix SQLBinaryOpExpr split bug & & modify wall config
added support for strong reference from OptimizedAssumption to OptimizedAssumptionDependencies
KAFKA - 9423 : Refine layout of configuration options on website and make individual settings directly linkable ( )
When _source is disabled , don ' t return it in realtime get fetching the document from the transaction log , closes .
[ scene2d ] ScrollPane now updates the widget ' s culling area in layout ( ) , not just draw ( ) .
ingest : support default pipeline through an alias ( )
[ GR - 10964 ] Move * Internal options to the same category ( DEBUG ) .
Polish " Debug mode is not logging web and sql related loggers "
SQL generation dialog UI fix
fill With InsertSetEncryptValueToken
Remove internal dep in LazyCloseable
Fix potential offset errors in BasicJsonParser
Use Versions . MATCH_ANY rather than NOT_SET now that NOT_SET is gone .
[ GR - 10706 ] Fixed merge issues .
Another ( almost pointless ) performance test
refactoring
Factor out different steps of dependency edge calculation from DependencyResolver # dependentNodeMap ( ) .
addressed @ bitiotic comment to use local variable
get non - capturing lambdas working
Fixed - CommandAsyncService blocks indefinitely if MapLoader ' s methods throw exception
Fix get / setReferentPointer
Set default target FPS to 61 .
fix a spelling mistake
BitScanReverseNode stamp tests should only be used with BitScanReverseNode
[ TEST ] NettyTransportMultiPortTests now checks selected random ports are free before using them
Disable granularity calculation per default
[ GR - 3599 ] LIRKind # merge is wrong with respect to derived pointer base .
[ fixed ] order of shutdown in AndroidApplication
Show ad markers after the window duration
Formatting
Formatting ( style / eclipse ) changes .
SimonStewart : Fix up some failing tests . Again .
Use ARGB color model in MatrixToImageConfig
Fix a gif recycling error .
[ TEST ] Use different name for remote cluster connection
Make Stopwatch ' s constructors package private .
[ GR - 13220 ] Introduced CompilerBackend option to remove backend - specific platform code .
VCC columns repacking fix
Allow Session ' s repository filter to be configured via the environment
Fix Drawee circles
Delete temporary blobs before creating index file ( )
Localization support for JDK - 9 .
JiayaoYu : Fixed the findElementsByPartialLinkText for firefox
Hive dialect name
[ inliner ] moved buildCallsiteHolderForElement ( ) to MethodInvocation
Remove topLevelConfigurationHook now that LIPO is gone
[ added ] Matrix4 JNI stuff to android build scripts .
Correct var names , improve documentation .
[ FLINK - 15113 ] [ config ] add fs . azure . account . key to list of sensitive configs
Fixed OioSctpChannel event loop issue by using 3 different selectors
Removed code duplication in Unicast perf tests for expected result .
Increase TruffleGraphMaxNodes to 15000
lucene 4 : caching should pass acceptDocs
[ JENKINS - 25338 ] Noting merge of .
Fix class name to be consistent with annotations
[ GEO ] fix pointsOnly bug for MULTIPOINT
Clean up warnings in the spring - test module
fixed poll - on - return for SPARC
CSV export : quote empty strings
Disallow rematerialization of non - JavaConstants
polish
Correct rewritting of script_score query ( )
Attribute binding updater
enable usage of ICU4J collator in ICU4J feature
fix to is apparently not complete .
Move all reads of Histograms to a single - threaded path . Either cache the commonly - accessed summaries at construction , or
[ TEST ] LoggingListener to restore the initial logger levels after any modification
Create " lastStable " symlink on the file system to point to the applicable build .
Polish CollectionFactoryTests
Condition changed to actually count to the number given as a parameter .
[ GR - 21695 ] Partial evaluation unfriendly methods on fast path in Agentscript .
[ GR - 26478 ] [ GR - 26507 ] Missing specialization and correct handling of destroyed objects after an optimized - > generic downgrade .
Fixup GWT Test runner
SimpAnnotationMethodMessageHandler skips template variable check in case of no pattern
Fix JmsTemplate default pubSubDomain setting
Code cleanup for PDF module ( )
Support array types in JavassistTypeParameterMatcherGenerator
Add @ Ignored Test case to reproduce SPR - 10243
security filter configuration is now in the war , so that it can be potentially customized by the user .
LinearScanAssignLocationsPhase : make extensible .
Add support for naive text alignment . ( )
Fix boxing warning
Worked on example and realm data plotting
HOTFIX : fix NPE in changelogger
[ FIXED JENKINS - 31649 ] Check should be against AccessControlled and Permission . READ
[ CID - 1182128 ] - Properly close InputStream in ConsoleCommand
Test server response headers
SimonStewart : More relable starting of the initial firefox profile . Using user . js for firefox specific settings
Adapted legacy Gson adapters into mini Gson .
Logging : BroadcastOperationAction - added trace logging for successful shard - level responses
Trace splitting summary implementation .
Simplify test
Fail to start if plugin tries broken onModule
Restore caching in OptionsParser . java
fixed testcase .
[ FIXED JENKINS - 27549 ] log job full name and catch Throwable ( + 1 squashed commit )
Convert FieldCapabilitiesResponse to a ToXContentObject . ( )
fix : use types with generics for overloaded method casts ( )
use YamlProxyRuleConfiguration . java
Fail with bad AppleConfiguration from Starlark instead of crashing .
[ fixed ] blue and green switched to display correct color
Make Optional . absent ( ) . hashCode ( ) different from Optional . of ( 0 ) . hashCode ( )
package fixes
[ hotfix ] [ test - stability ] Properly fail if ZooKeeperTestEnvironment cannot delete ZNodes
Allow CppCompilationHelper to disable coverage instrumentation
Add a check for current libc . Add check folding
Fixed perfect reflection on collision
fix loading of OptionDescriptors providers on jdk9 + to avoid duplicates
[ hotfix ] [ runtime ] Minor optimization in CheckpointMetrics
HLS : Fix slow seeking into long MP3 segments
[ FLINK - 3122 ] [ Gelly ] Generic vertex value in label propagation
Small fix to CompressionLoader
Revert " escape special chars on status message "
Minor style changes ; optimized nested loops at end to avoid some redundant computation
Rename AllocatorTest . test to testAllocation to avoid confusion with GraalCompilerTest . test .
Multiply use shift - add | subtract operations when easily possible . In DivNode use rightshift
bug fixed for WriteBigDecimalAsPlain , for issue
Changes after @ imotov review .
rename to InsertSetQueryAndPlainColumnsToken
refactored String related functional tests out in a separate test class .
[ GR - 13153 ] Ensure jdk_libs are available in lib / svm / clibraries / < platform > .
Expect correct data type in interop test .
Use assertThrows ( )
Update version for signing scrolls after backport
for , refactor VisitorRegistry
Lazy compute SSLSession creation time .
[ FLINK - 1633 ] [ gelly ] some cosmetic changes to getTriplets and EuclideanGraphExample
Removed unnecessary bit operation from MemorySegment
refactor DatabaseUtil
for , refactor ShardingInsertOptimizeEngine & EncryptInsertOptimizeEngine
Add ability to allocate memory for an object without the use of the default constructor .
[ FLINK - 17617 ] Fix SQL Client autocomplete
[ TEST ] Make sure that the facet fields exist .
[ hotfix ] [ tests ] fix codestyle issues in NettyShuffleDescriptorBuilder
Add hikari custom configuration support ( )
Moved tests over from testsuite to core
Bump release to 2 . 10 . 0 and update release notes
Tidying up HealthReport code to use new utility methods HealthReport . min and HealthReport . max
Use MediaType . APPLICATION_CBOR constant
[ fixed ] synch issue for AndroidMusic . Me stupid . . .
KAFKA - 5422 ; Handle multiple transitions to ABORTABLE_ERROR correctly
RedisCommand regression fixed
Ignore warnings related to types deprecation in REST tests . ( )
Consider dash & underscore when equating indexed elements
Cleanup - docs
[ SPARC ] Updating jump targets for fcmp instructions when doing unordered compare of float / double values
Cleaner implementation of server detecting client disconnection in sampling stream
[ FLINK - 13407 ] [ tests ] [ coordination ] Harden StandaloneResourceManagerTest
table DDL refactoring
GT - 2856 Corrected restore project NPE observed when using large
[ GR - 11566 ] Added log options into help output .
Minor improvements in the local and the queue scheduler
Actually rebuild the axes at the beginning of the matrix build
Improve test connection
refactor BindingTableRuleTest
[ ] Cleanup
Fix host - port splitter in SentinelConnectionManager for ipv6 addresses
Cleanup Rescorer Tests & add random indexing
Fixes CuratorZookeeperRepositoryTest test errors ( )
Read first bytes as ints to respect method contract
fix formatting in MonitorSnippets
Add cause to assert_no_failure when replay translog ( )
remove unused code
Fixed - Redis Password visible with TRACE log level
fix : read correct buffer size for string pool parsing ( )
store the doc type in the internal index ( )
Ensure people are aware recycler capacity is per thread .
Clock skew correction is not propagated to more than one level of local span . ( )
[ FLINK - 8559 ] [ RocksDB ] Release resources if snapshot operation fails
Replace pendingFormat with waitingForFirstSampleInFormat in SimpleDecoderVideoRenderer
Update Execute Watch to allow unknown fields ( )
[ TEST ] Replaced File . separator with ' / ' as REST test suites can be loaded from classpath , where ' / ' is needed
JAVA - 943 : Migrate spring - mobile to parent - boot - 2
remote : rename auth flags .
[ FIXED JENKINS - 9326 ] Jenkins Maven build does not recognize Tycho surfire reports
[ Fix ] - access granted on internal calls should be logged on trace
Fixed race condition in UnilateralSortMerger shutdown .
Always omit method return types in TypeConfiguration to avoid duplicates .
Polish MultipartBodyBuilder
Fix crash in AvatarImageView .
Polish " Remove unnecessary config options for metrics "
Complete ArchiveController # listMonthArchives
HttpsServer can use TLSv1 . 3 on JDK16 + ( )
Adding wait for a new window to open
Oops fixed bug in guessing Shift_JIS - - flipped if condition !
make sure to pass the updated settings from the plugins to the modules with transport client
change visibility
Pass through a " cpu : " tag to the execution info of genrules and tests .
SQL context informer ( hyperlinks , auto - complete ) cache fix ( use ds listeners )
merge
[ changed ] Eclipse formatter on / off tags to @ on / @ off .
Fix style and isLocalOrAnonymousClass
[ hotfix ] [ tests ] Reset state to allow retry on failure
Changed one line where I hadn ' t used the same brace - placement as the rest of the code .
Java api : unify SearchResponse and BroadcastOperationResponse code around shards header
Restore proper use of CacheLoader
[ FIXED JENKINS - 18403 ] Detect when an attempt is made to run Maven on JDK 5 .
Add an option to enable an experimental UI
Fixed wrong boolean expression in determining whether to skip the pool or not
fix logging
fix for IfNode . pushNodesThroughIf ( push more than one node )
Warn about using multiple platforms with " - - platforms " , but do not crash
Allow to offload certificate validation when using BoringSSL ( )
Switch HSTS enum to upper case .
BAEL - 4225 Thymeleaf Variables
Fix version parser used by gradle ( )
Add failing test for concrete @ RequestMapping method in abstract superclass
merge
Fix ` value ` method on ip scripts ( )
EQL : Add IsNull / IsNotNull checks ( )
Handle classpath entries with wildcards in NativeImage . processClasspathNativeImageProperties
Remove unnecessary field
Fix ObjectTreePrinter .
[ FLINK - 7998 ] [ examples ] Fix TPCHQuery3 examples
Update TtmlDecoder to keep only one Span of each type
[ ML ] Rename ` xpack . ml . allocation_enabled ` boolean setting to ` node . ml ` .
rename to getColumnIndex ( )
Fix invalid boundary in generated dynamic dispatch cast .
Added null check for compilation profile .
gui : cache renderer results in find / usage dialogs
replaced magic constants with GraalOptions . CheckcastMaxHints and GraalOptions . InstanceOfMaxHints
Minor DefaultDrmSessionManager tweaks
Move the query output formatters to ServerBuilder .
Support JsonReader API on DOM objects .
Add utility methods for network type and country .
clean
DrawableLoader - > ResourceLoader
clear all pixels when dispose is 3 but previousImage is null ( )
MINOR : cleanup RocksDBStore tests ( )
Add some assertion messages
Switch some x - pack tests to new style Requests ( )
[ FIXED JENKINS - 18356 ] Restored deprecated AbstractProject . doBuild { , WithParameters } overloads .
Fix automatic android sdk opening on MacOS , actually add images on generation
add initialDelay
fixed testcase failed
Extends HttpPostRequestEncoder to support all methods except TRACE
WAV : Don ' t read past data end position
[ GR - 18577 ] TruffleFile API is inconsitent wrt exits and getCanonicalFile .
Fix single context assumption was not invalidated properly with shared engines .
Network type cleanup
[ ROCKETMQ - 47 ] Avoid broker updates of NameServer address list on startup twice , closes apache / incubator - rocketmq
Fix array initialization
Show column description in header ( rendering )
Fix HLS ID3 sniffing
Editor refresh on save : force refresh only for new objects
Fix webm demo
[ FLINK - 1201 ] [ gelly ] replaced create with fromDataSet
refactor scanNChars
enhance datasource monitor page
Use BatchErrorHandler when Kafka listener type is batch
Shuffle the list so all clients don ' t prefer the same sentinel
Remove afterUniverseCreation handler .
Snapshot / Restore : restore of indices that are only partially available in the cluster
No need to create to extra Point objects only to pass their fields to new Path object .
HandlerMethod caches interface parameter annotations
Mustache : Ensure internal scope extrators are always operating on a Map
[ FLINK - 11594 ] [ tests ] Add assert to test that TM re - connects to JM
[ hotfix ] Fix checkstyle violations in AkkaRpcActorTest
[ ML ] Improve message when native controller cannot connect ( )
Add ExpectedExceptions for proxy
[ hotfix ] [ e2e ] Add ' flink ' prefix to flink log backup directory
[ FLINK - 11046 ] [ elasticsearch ] Always re - index buffered failed requests on snapshot
Fixed getting signature if returnType is null .
Add UnsupportedMediaCrypto
Use line endings on all OSs when writing image config JSON
[ ML ] Refactor method to process model size stats ( elastic / x - pack - elasticsearch )
Upgrade to Elasticsearch 5 . 4 . 0
[ ] Ensure correct cause of handshake error is included in the SSLHandshakeException when using OpenSslEngine .
optimize code , remove not used global variable .
[ fixed ] Vertex arrays on the desktop .
1 . 2 . 11 SNAPSHOT
Remove FIXME from JPA sample
Deprecate the no - jdk distributions ( )
, fix test case
refactor ConnectionIdGeneratorTest
[ FLINK - 5945 ] [ core ] Close function in OuterJoinOperatorBase # executeOnCollections
Removed Actor # setCenterPosition , added setPosition ( x , y , align ) .
Fix SelectTelnetHandler . telnet IndexOutOfBoundsException
[ added ] small change in UITest for regression .
Clean up SnappyIntegrationTest / Fix checksum encoding
[ FLINK - 19933 ] [ DataStream ] Execute and collect with limit fails on bounded datastream jobs
Ensure the tests complete on java7 and java9 as well .
bugfix : NettyPoolKey toJsonString error ( )
HystrixContextScheduler was not wrapping the Inner Scheduler
[ ] Stop flush if one outboundbuffer was freed and do the same for inboundBufferUpdated if an inboundbuffer was freed
Reorder code to keep related values together
[ GR - 16402 ] Remove workaround of incorrect subword conversion in InvokeCFunctionPointer .
Fixed - Blocking commands connected to Redis Cluster aren ' t resubscribed after Master node failover .
[ flow - sensitive ] readability , baseCaseIsNullNode
Elasticsearch 7 . 5 is on Lucene 8 . 3 . ( )
Started to implement blocking in - memory channels
source not returned when * specified in fields list , closes .
Provide control over how a Cassandra Cluster is created
Use instrumentId instead of instrument as parameter name for getInstrument .
CcProtoLibrary : Don ' t add dynamic librarys to filesToBuild on Windows
change method to return Optionals ( )
Fix seeking to a non - loaded period .
[ GR - 25913 ] Add option to disable type id result verification .
Remove unnecessary resolves .
[ GR - 21918 ] AssertionError in Context . asValue after Context has been closed .
Remove Temporary Timeout in take
Fix Snapshot Completion Listener Lost on Master Failover ( )
Fix failing logging listener tests
Incorrect code rejected by JDK 7 ' s javac ( and NB ' s editor ) .
[ FLINK - 10655 ] [ rpc ] fix RemoteRpcInvocation not overwriting ObjectInputStream ' s ClassNotFoundException
- Changed leak canary toggle text to " monitor leaks " .
SimonStewart : Enable browser - side logging if the debug flag is set .
SimonStewart : Making sure we actually wait long enough
[ ] Fix handling of chunked requests in HttpPostRequestEncoder and the multipart examples
Move DefaultLoadErrorHandling creation to DefaultDrmSessionManager
getParameterSets ( final StatementExecuteUnit executeUnit )
Clearer diagnosis for JENKINS - 12543 and similar issues .
Update the version compatibility for zero_terms_query in match_phrase .
Enable pending test elimination for non - integer stamps .
[ TEST ] Fix indexed script test .
Add identity comparison fast path to keyEquals .
[ FLINK - 4021 ] [ network ] Consume staged buffers on shutdown
More small improvements to SKIP_N_BARCODES - - now remembers value across invocations of doDecode ( )
Fix NPE in Logfile Audit Filter ( )
More JavaDoc
Link to tablespace
Minor cleanup to LotteAnimatableColorValue
MySQL connection page redesign
improve web sessionStat
Downgrade to JDK6 . remove ( Object , Object ) is not existed in HashMap of JDK6 .
[ FLINK - 16836 ] Clear rpcConnection field in JobManagerLeaderListener when target loses leadership
Fix license expiry log message
isShuttingDown check fixed
Driver libraries resolution fix
, support fetch size in DatabaseMetaDataResultSet ( )
resolve merge conflicts of fc349fd483 to master .
Add EPoll support under Java 11
[ FLINK - 9841 ] [ rest ] Close log file channel after response was fully written
allow to filter also by node _name and _id , make sure to reroute properly after cluster update settings
Update tests to align with changes made in 8094076
Remove _type argument from IngestDocument after merging in master
Remove unnecessary instance variable in QueryStringQueryParser ( )
KAFKA - 5624 ; Add expiry check to sensor . add ( ) methods ( )
modify the config reload
for , refactor createEncryptCondition for ShardingOrPredicateFiller
Fix confusing variable naming in toolchain launchers .
Fix potential problems creating a test driver
jtds xa support 1 . 2 . x & 1 . 3 . 0
Fix skylark aspect issues .
[ FIXED JENKINS - 15465 ]
[ GR - 3244 ] Load all non simm13 constants for further OP3 ops .
CheckCastNode . canonical should not assume that tool . assumptions is not null
Pull reading of data bytes back above if ( . . . ) { continue } clauses
Use TruffleLoggers instead of System . out in TRegex
[ CACHE ] Use correct number of bytes in query cache accounting
Expose default RabbitMQ credentials in the metadata
[ FIXED HUDSON - 6435 ] move " Checking cyclic dependencies " step to PLUGINS_PREPARED
Automatic code cleanup .
recover fatal error
Avoid recursive wrapping by unpacking TruffleList and TruffleMap when converting back to guest language object .
Fixed compile error and some typo
core : fix help for jadx - gui , improve code
Expose local port in EmbeddedServletContainer
Avoid image pulls in integration tests
Added ConnectionlessBootstrap for UDP transports
Don ' t set uncaught exception handler if it has been set already .
Truffle - DSL : fixed NPE if wrong invalid child type is declared .
bwc : enable script cache in node stats ( )
SQL formatter fix . Loggign added for PK cache
[ GR - 15111 ] Use canonical paths in tests .
[ GR - 14593 ] IllegalArgumentException thrown when attaching VS Code to R terminal .
no comments
Handle missing actions when parsing WatchStatus
Update webrtc to M68
Watcher : Ensure watch status needs to be udpated on unmet condition ( elastic / elasticsearch )
SPR - 6786 : added more checks for quality factors & charsets
Simple fix to respect return value return by File APIs .
Assertj refactor ( )
case mapping tables with quotes fixed
Fix results not being persisted ( elastic / elasticsearch )
ManagementLinkTest compiling against HtmlUnit 2 . 17
Revert " removed double javadoc "
better logging for shard and index
Remove TODO in MaxSizeCondition ( )
test fixes
BAEL - 3242 : Use random port for test execution in spring - vertx module
Refactor CommitId examples
Fixed issue : NETTY - 325 ( QueryStringDecoder doesn ' t properly handle missing query string values and other corner cases )
test : ensure that no percolate request are executed when stopping a node
PostgreSQL data types cache fix
Moving the location of the mobile - install launch script .
[ GR - 25049 ] UnsafeSubstitutionsTest stores object to int field .
[ FLINK - 2448 ] [ tests ] Create new Test Environments on every call to getExecutionEnvironment and allow sharing of job execution results
Introduce a type mapping
[ FIXED JENKINS - 13546 ]
Fix PreBuiltTransportClientTests to run and pass
Address Devin ' s comment
EranMes : Making the extra environment visible .
added Log . printObject ( ) and Log . printlnObject ( ) for more detailed logging of objects
Fix issue with stale createPeriod events in ConcatenatingMediaSource .
[ GR - 10553 ] Fixed formatting .
test coverage for DataSourceConverterTest
Fix stuck playback when media has uneven track end times
[ GR - 14072 ] Fix shape memory leak .
delete InsertType
add conditions for check email
Docs : ctx : Add more cross - reference links
[ Grid ] Tests - throws is not needed [ skip ci ]
ERD : bring to front & send to back actions
[ DatePicker ] Made some MaterialCalendar related methods / classes public , but still restricted .
Use a MatchResult created by a successful match , as required by Harmony ( used on Android ) :
[ JENKINS - 34675 ] - Fix the Unit test
[ fixed ] Major bug with sprite rotation .
Missing one xAxis entry
add MySQLPacketStatementExecuteQueryResult
Update core - java - modules / core - java - lang - 3 / src / test / java / com / baeldung / checkclassexistence / CheckClassExistenceUnitTest . java
TraceRA : reuse assignment procedure .
Fix compile issue in compiler suite .
Guard MQVesion methods .
fix line len
Properly handle bazel ' s output deletion within a write - protected directory .
Update ImageLayer . java ( )
Ensure HlsMediaPeriod works with playlists without variants .
RSV : use connection recovery for data save
[ TEST ] Fix FieldSortIT failures
[ FLINK - 1287 ] LocalizableSplitAssigner prefers splits with less degrees of freedom
inherit from the parent .
1 . 2 . 40 - SNAPSHOT
[ TEST ] Reduce number of buckets created in InternalDateHistogramTests
Update Downsampler names .
Clarify wording for the strict REST params message
inputstream retrieval added
use display name instead of ID for cause
BottomAppBar change interpolation to be 0 if fab is not visible
Allow partial format for adaptation check
Make ConcatentatingMediaSource provide a dummy media item
Add test for calling functions from dynamically instantiated modules .
: Set the correct bwc version after backport to 6 . 0
Improve knowledge of last playing period in AnalyticsCollector .
Fix CLI test failures
FS Gateway : Allow to configure is native file copying will be used , closes .
Scrolling issues , fix releasing search context eagerly , they should not be released when scrolling , closes .
Add support for CustomConversions with Cassandra
Fix failure exception message report back to client
Remove use of ` import static ` .
DanielWagnerHall : L ' et ' s try this Sauce - recovery again . Where exceptions are actually exposed .
Avoid unnecessary Node array allocations .
KAFKA - 5959 ; Fix NPE in Sender . canRetry when idempotence is not enabled
Starlark : specialize StarlarkInt . shift * for longs
Allow overwriting table elements from the linker .
Delete ctx . file_action
MINOR : Safe string conversion to avoid NPEs
[ JENKINS - 8711 ] Post build action deploy to maven repository can fail using " private maven repository " option
support getSchema
Add Format field to AudioSink . InitializationException
Removed redundant code
[ FLINK - 16291 ] [ hive ] Ban count from HiveModule
Hide " My contacts " phone number privacy option .
Fix wrong URL encoding in watcher HTTP client ( )
Do not load PROVIDED dependencies
Dump inlined call tree only if inlining happened .
Set timestamps of nested jars
Pass Assumptions to CanonicalizerPhase to avoid NullPointerException
HTTP / 2 DataCompressionHttp2Test test complete race condition
Reformatting try - with - resources
statically reference the plugin and mark as experimental
[ FIXED JENKINS - 31487 ] ( )
[ ] Respect IOV_MAX when call writev in native transport
Switch must work for any condition type .
- Fixed playlist fragment infinite update cycle .
[ hotfix ] [ REST ] [ docs ] Add missing spaces
Rolling back changes to override Chip # setText and Chip # getText because it broke some tests .
TraceRA : TraceLinearScanWalker : clean up splitBeforeUsage and implement findOptimalSplitPos .
ConnectionWatchdog throws exception when eventloop in shutdown state
handle emoji density scaling more correctly
Removed json ( ) , html ( ) and xml ( ) predicates
Fix invokeBasic invoke mechanism
Consistently check if AsyncContext already completed
Remove deprecated wakelock from AndroidDaydream
Canonicalize additional cases in IsNullNode earlier .
Task last run format fix ( NPE )
AArch64 : jaotc generates incorrect code for compressed OOPs with non - zero heap base
Tree , enable overriding expandAll .
ScrollPane , oops commented wrong line .
Better error handling and fix multi - language shell .
[ FLINK - 18058 ] [ mesos ] [ tests ] Increase heartbeat interval / timeout
modify the cross anazlyer
Improve ` UserInfoTokenServices ` logging
Geo : Allow to parse lat / lon as strings and coerce them
throw exception instead of return max dop of 1 one IOException ( from file system / hdfs )
[ GR - 3893 ] Implementation of MXGraalBean . setAttribute .
Fix a few style issues .
remove no longer necessary reflection
1 . 1 . 17
bug fixed
AbstractRequestLoggingFilter ignores non - available query string
Polish interceptor registration MVC config
move persistenting state and sharsd state in local gateway to another thread
[ streaming ] KafkaSource checkpointing rework for new interfaces
[ TEST ] Verify if clear cache request went to all shards .
[ FLINK - 6365 ] [ kinesis ] Adapt default values of the Kinesis connector
Provide a method to retrieve a closeable char [ ] from a SecureString ( )
Centralize the aapt version logic to a single method .
Renamed invalidate into invalidateOSRTarget to make the methods behavior explicit .
Make FsBlobContainer Listing Resilient to Concurrent Modifications ( )
xml error location
[ hotfix ] Add MetricUtils # startMetricsActorSystem
[ TEST ] Increase ML distributed test job open timeout ( )
Added AzureBlobStoreContainer tests
Eliminate references to Joiner in toString static utilities , inlining the joining logic explicitly .
Removed NodeLIRGenerator again .
[ tracing ] : Ensure tracers are found from - - ext classpath
Keypad buttons added
minor optimization by Keith Wong
Expand ios_test to accept xctest_app targets that provide an XcTestAppProvider .
Oracle ROWNUM limit transformer
Fix trailing slash issue
Polish
[ TEST ] use randomPositiveLong where possible
Avoid the use of lambdas in tests .
Add trace logging when aquiring and releasing operation locks for replication requests
Re - enabled tests for meta - annotations on interfaces
Throw a PrematureChannelClosureException if the channel was closed before all responses were received for the sent requests . See
resolve store the language as normalised attribute so we can access it from the track parser
[ FLINK - 19377 ] Change parameters of EventTimeWindowCheckpointingITCase
Avoid running cleared requests in ThumbnailRequestCoordinator .
adjust inlining policy ' s maximum trivial size
for checkstyle
MINOR : Make info logs for KafkaConsumer a bit more verbose ( )
Discovery : add a finalize round to multicast pinging
Use object identity when comparing AST nodes in ASTTransition
[ GR - 20435 ] Only major version is used from JAVA_VERSION release file entry .
[ grid ] Disabling TRACE method in hub
[ FLINK - 5632 ] [ streaming api ] Fix typo in StreamGraph variable name
fb4a - aladdin : fix deferred releaser concurrent impl - releasing in FG thread might cause race condition
Consistent number handling in StandardTypeComparator ( BigInteger support , doubleValue fallback )
[ GR - 8492 ] ThreadFactory implemented to mark threads as daemon .
fix code review .
Commit / rollback commands are enabled always ( in manual txn mode )
TypeSwitchNode should be more careful about deleting successors
Fix test assertion matching exception message
[ hotfix ] [ task ] Generalizing MailboxExecutor # execute to ThrowingRunnable .
ensure instanceof and null check stay dependent
test regression
small fixes and formatting work
[ GR - 15208 ] Add assertion to avoid duplicate language ids on the classpath .
[ FLINK - 1574 ] Fix null RuntimeContext when opening combiner UDF in CombiningUnilateralSortMerger .
Bootstrap check for OnOutOfMemoryError and seccomp
Make ConnectivityOptions available to all commands .
resolve get language for all types , not just audio
[ FLINK - 19305 ] [ coordination ] [ CI ] Setup CI run with declarative slot management
NETWORKING : Fix IpFiltering Test ( )
work on connection management
Adding annotations to BottomSheetBehavior
Testing : Ensure http . enabled is set consistently for all nodes in test
Bael 3893 - Code review updates
Change default order of AutoConfigureOrder to 0
modify sql size
Check for validity before usage .
Small fixes to subsampling layer ( )
[ JENKINS - 48775 ] HACKY HACKY FIX
RSV auto - refresh - NPE fix
BAEL - 3975 : Fix the PactVerification test ( )
Set Application Name on Connect
Change dst to dest
Update desugar to fix method signatures after adding emulated interfaces
1 . 1 . 2 - SNAPSHOT
[ DOCS ] : [ fix comment statement and reformat code ] ( )
Open - source J2ObjcConfiguration and J2ObjcCommandLineOptions .
2 . x : Fix TrampolineScheduler not calling RxJavaPlugins . onSchedule ( ) , add tests for all schedulers . ( )
DruidDataSource support failFast
Use equals ( ) to compare ResolvedJavaType
SQL : Fix the name of the indices lookup action ( elastic / x - pack - elasticsearch )
Fix for @ Options annotation and generated keys
[ FIXED JENKINS - 31954 ] Stack overflow reconfiguring some settings where Descriptor . newInstance is gratuitously overridden .
added logging for failure to create columns
KAFKA - 10755 : Should consider commit latency when computing next commit timestamp ( )
Don ' t save local data using DiskCacheStrategy # ALL .
Create ForeignAccess lazily
Split up killCFGInner
Read trigger definition from database
Fix DownloadHelper for some HLS streams
SerializerFeature . SkipTransientField support java . beans . Transient , issue
Remove version check after backport ( )
[ GR - 13701 ] Thread - Safe Lazy Parsing .
modify createEncryptRuleConfiguration ( )
Fix MockCookie parsing when attribute is omitted
Internal : fix shard state tranport action names
Fix store transition to multiple threads failed from state constant and dynamic .
JasonLeyba : Properly encode null values in BeanToJsonConverter . Fixes issue 1133 .
for , refactor QueryOptimizeEngine
Aggregations Refactor : Refactor Cardinality Aggregation
Minor fix of _cat output ( ) ( )
Allow serverStdout to send PORT_LOG_MESSAGE in any of the first 12 lines
Only search for visible parts of a contact .
Marked 1 . 4 . 3 as released
Update scripts for Metropolis project
Generated DDL comment
Fixes
don ' t put spaces around property separator ( | )
- More bug fixes
Fixed unquoted values containing a foward slash .
eliminate old 1 . 7 signatures for CBC methods
Take advantage of buffering in HttpSnoopServer
Add support for the ' 0o ' octal prefix for integers .
Fixed a deadlock .
Fix race condition when a child finish its deployment after its parent deployment failed
Add " force download " option
compute the hitrate
Add option to populate selector list in blaze query ' s proto output .
Add YamlShadowRuleSpringBootConfiguration
Added version support in RestClient . Revised CallPath to extract version number specified in the path .
Use comparator for Reconfigurator ( )
GT - 3579 - Fixed type casts
Polish
Truffle : Move direct call logic from call site to optimized call target .
Plaintext presentation fix
[ JENKINS - 17374 JENKINS - 18116 ] Don ' t set gzip header for error
bug fixed for getRemoteAddress .
fix send duplicate jvm data issue
Adding support for formatter options
Not cause busy loop when interrupt Thread of NioEventLoop
tests refactoring
[ JENKINS - 27392 ] SimpleBuildWrapper . createLoggerDecorator
[ refactor ] TraceRA [ BU ] : split loop back - edge handling and forward edges
MSSQL : datetime columns scale in DDL fix
[ fix ] Conditional Elimination : Always improve pi stamp if new guard provides additional information .
Fix bitmap reuse bug when cross fading .
@ Lazy falls back to empty map / list / set for non - required collection dependency
fixed up stuff
move to 0 . 90 . 0 . Beta2 snap
more bug fixes .
Fix MeshBuilder # box
TEST : Corrupt some translog files used in recovery ( )
painless : Small cleanup of Debugger class to use StringWriter
Aggregations Refactor : Refactor Filters Aggregation
for : refactor DDLAssertEngine
KAFKA - 9140 : Also reset join future when generation was reset in order to re - join ( )
Fix test
Fixed using same generator for multiple fonts with different sizes .
fix : allow to regenerate class code ( )
Insert a placeholder state for intrinsics that contain an invalid frame state
Fix checkstyle line length limit
Fixing test to run on any platform ( IE is available on Windows only )
for checkstyle
Fixed iterator not been converted
Revert " Revert " Merge pull request from s1monw / issues / 17090 " "
Code cleanup
[ FIX JENKINS - 34093 ] Make logger field private
Added new assignment strategy for arriving instances which works better with lazy task assignment
[ FLINK - 18821 ] [ network ] Report PartitionRequest failures to subsequent callers
[ GR - 3555 ] Use UNSAFE malloc / free / memcopy in Sulong instead of a native call via TNI .
Avoid making stubs for dead switch successors
Sulong : change equalsIgnoreCase to equals for start and context initialize and dispose function look up .
add test case
fixed accidental change to byte array loop
Fixed merge residue
Fix analyze / lint errors
Use SuppressWarnings on methods instead of classes .
QM code cleanup
[ FLINK - 8488 ] [ flip6 ] Fix Dispatcher job recovery bug
Minor improvements
Fixing number of default node configuration capabilities
objc protos : Pass up dep on C + + runtime if depending on C + + API
polishing
- Modified recovery to not set if progress position is 0 or less .
finishing up redirect work
Update variable name in comment .
re - instate synchronized which I reverted in a previous commit
enhance class parse
Improve stability of the shardsCleanup test
buildjar : close streams
Add ' . ldscript ' as a possible linker script file extension .
modify the jsp and filter
Fix for NETTY - 150 CompactObjectInputStream web start ( applet ) Class . forName issue
[ FLINK - 2097 ] temporarily disable session management API
Use String . CASE_INSENSITIVE_ORDER instead of custom Comparator
use throw new ShardingException ( ex ) ;
remove try in posIndex
[ SPR - 8030 ] AbstractJUnit38SpringContextTests and AbstractTransactionalJUnit38SpringContextTests are now officially deprecated .
Bug fix , NetClient worker losing message when buffer arrives before the connection is added to the context map
move construction of ConfigChangeEvent outside the lambda expression ( )
Query DSL : Remove duplicate code in AndQueryParser
Get rid of ui . forms
Scripting : Add back params . _source access in scripted metric aggs ( )
Add breakdown to sandboxing preparation / overhead to JSON profile .
Allow clipping offset windows
2 . x : Improve Completable . delay operator internals ( )
JasonLeyba : ChromeDriverService # isRunnig ( ) ' s return values were inverted .
sulong : add null checks for location logging
[ FLINK - 5326 ] [ network ] Check release flag of parent in reader
[ TEST ] Add more engine unittests for sync IDs
[ hotfix ] [ tests ] Remove unnecessary try - catch usage
Fixed NETTY - 381 channelDisconnected event is sometimes not triggered when Channel . close ( ) is called by multiple threads .
Data stream support for ILM remove policy API ( )
Change video Id
SL : migrate SL to use @ GenerateNodeFactory .
Fix a number of test failures running the backport under JDK5 .
ServerWebSocket asynchrone handshake method should consume a future instead of a promise - see
Use Windows workaround for missing bool type for all query code generation
Simplify the acquisition of Cleaner
fixed the dependency computation in MatrixProject .
- optimizer nodes hold key fields in FieldList to preserve order
SQL Server references read fix
DragAndDrop , proper target hit detection .
ORacle : use DB_ view to read tables and columns ( if enabled )
Correct errors - HRLC TermVectorRequest ( )
[ tinker ] FIX : build failure .
Add possibility to write extractor dump files to device
Robustness fix for malformed ID3 metadata
[ hotfix ] [ rpc ] Add RpcConnectionTest to validate that connection buildup fails fast when endpoint is unreachable .
GWT : do not use Flash for sounds by default ( )
Update conditions for PIN megaphone .
[ FLINK - 10819 ] Replace with the Flink ' s Deadline implementation
Prevent error when user has no email client .
Add stack size to execution state .
KAFKA - 8774 : Regex can be found anywhere in config value ( )
Cleanup in tests .
[ TEST ] First decrement in - flight requests before releasing the latch
Remove connection read / write timeouts for web sockets .
[ FIXED JENKINS - 12680 ] the onOffline method wasn ' t called for abrupt client termination
improved mysql parser support . for interval function
improved construction of query string .
Fixed - PingConnectionHandler throws CancellationException .
[ FIXED HUDSON - 7256 ] use setContext ( ) instead of getContext ( ) . setAuthentication ( )
[ TEST ] Don ' t pass possibly oversized array with 0 padding to SearchRequestBuilder # setFacet ( byte [ ] )
Refactor handling for bad default permissions
Revisit AuditEventRepository interface
[ FLINK - 12223 ] [ Runtime ] HeapMemorySegment . getArray should return null after being freed
report proxy exception in ChannelInboundHandlerAdapter , add more unit tests for error in proxy
fixed for : createEntrySpan should get SamplingService only if it required ( carrier is valid ) ( )
Move TODO into a bug .
[ java ] Resetting the port to zero after using a service builder to allow reusing the same Builder
Expand check for muxed audio media tags to include uris that match variants
Test : force merge index in the end of IndexStatsTests . throttleStats
[ hotfix ] [ config ] [ docs ] Fiy typo for ` taskmanager . memory . preallocate `
change phase order ( ConvertDeoptimizeToGuard earlier , PEA after loop peeling / unswitching )
Fix missing MasterSecret in PartProvider
[ GR - 15111 ] Use canonical paths in tests .
Upgrade to Netty 4 . 1 . 52 . Final
Avoid double - recovery when state recovery delayed
Make sure runtime shuts down on exception in uncaught exception handler
Fix range of version number generation in test ( )
Add support for registering custom MIME types
fix bug when load album is null
allow InterruptedException so that test code that extends from it can throw InterruptedException as well .
Prevent testing on double values whose toString may use the scientific notation .
Test : reduce load in RecoveryWhileUnderLoadTests
update version to 1 . 1 . 15
Fix extraction of PCM ( sowt ) in MP4 / MOV
Tweaks from shuffling of internal hash functions .
Support removal of ranges from ShuffleOrders
[ FIXED JENKINS - 34883 ] make method protected
Calling cancel on a Future returned by a TaskScheduler works reliably now
Supported relaxed binding on inner classes
merge fixes
remote : Allow auth scopes to be a comma - separated list .
KAFKA - 6914 ; Set parent classloader of DelegatingClassLoader same as the worker ' s ( )
PG schema contents loading fix
delete execute units
[ hotfix ] [ rocksdb ] Convert performance benchmarks to unit tests
Skylark : proper anchor is generated for Boolean types in the documentation .
[ ML ] Fix exception categorising an empty string ( elastic / x - pack - elasticsearch )
Checkbox tree update fix
Remove unnecessary text selection refresh
Mark field in ReindexSameIndexTests as final
[ TEST ] pre check download service working
Typos .
Issues / 3593 ( )
Use single locale - specific calendar to read date / time values
[ aquery ] Fix the case with duplicated Target entries .
Add unescapeCsvFields to parse a CSV line and implement CombinedHttpHeaders . getAll
Fix interpolation for MSD round rects
Put module - info into a place required by multi - release jars
Introduce TraceTrufflePerformanceWarnings flag .
Remove initial seek counting in ExoPlayerImplInternal .
Reenabled FileInputSplitAssigner as default assigner for FileInputSplits
[ FIXED JENKINS - 44361 ] Follow HTTP redirects while initiating CLI connection
[ GR - 13233 ] HotSpotReplacementsUtil . methodHolderClass should be folded .
Get the status servlet working again .
[ FLINK - 18552 ] [ tests ] Update migration tests of TypeSerializerSnapshotMigrationITCase cover migration from 1 . 10
Make GuardProxyNode ready for the case where its input is null .
Sort contacts without names after contacts with names .
Correctly propagate format identifier for CEA - 608 in HLS
Fix ' spring . resources . cache . period ' for WebMvc
painless : optimize / simplify dynamic field and method access
Resolve same method definition brought by merge
OM DSL : use setShapeAndGrow ( ) when only changing the ObjectType
Trogdor : Fix / coordinator / tasks parameters to accept long values ( )
OK , issue 303 , work around different Behold 2 setting
SimonStewart : Cleaning up some compilation warnings
Remove unnecessary recipient resolves .
add assertGetAggregationDistinctColumnIndexes ( )
Including method . hashCode in the hashCode of NodeSourcePosition
Fix GitHub by reimplementing Cartesian { Set , List } . contains .
Having two neighbors pointing to each other is important to make RunMap . removeValue work .
Use the C standard library ' s ` memmove ` instead of a naive implementation
Improve handling of touch input for Slider with multiple thumbs
Ensure that Jetty fails to start when its thread pool is misconfigured
[ FLINK - 18137 ] Handle discarding of triggering checkpoint correctly
[ TEST ] Enable test compilation against elasticsearch version 1 . 5 . 2
Fix MCR comment line break .
LSRA spill optimization : insert spill moves eagerly .
fixed possible NPE . installations can be null
KAFKA - 3147 ; Memory records is not writable in MirrorMaker
Object shouldn ' t be tested for equality with itself . This was identified using Error Prone check .
[ Fixed JENKINS - 8614 ] ProcessTreeTest fails on Windows
Transformers menu
Use CustomCommandLine directly instead of via SpawnAction . Builder .
[ FLINK - 18904 ] [ hotfix ] Sort methods in StreamConfig
Process OP_WRITE before OP_READ to free memory faster
Removed listen parameter from SubmitJobDetached message .
PG SQL hyperlinks - show function signature
[ FLINK - 19669 ] [ coordination ] PipelinedRegionSchedulingStrategy # init ResultPartitionType blocking check use isBlocking method
KAFKA - 9919 : Add logging to KafkaBasedLog : : readToLogEnd ( )
Add ForwardedHeaderFilter requestOnly
Fix infinite calls to executeAndSpecialize when using implicit casts with caches and only implicit casts updates .
Fixing use of deprecated method
assertCannotRegistryResourceAgainWhenDataSourceIsNotClose = > assertFailedInXAResourceUnReleased .
[ FLINK - 15134 ] [ client ] Delete temporary files created in YarnClusterDescriptor
[ GR - 15180 ] NullPointerException from NodeUtil . getNodeFieldName .
Executable instead of Object for constructor or factory method
[ FLINK - 5201 ] [ logging ] Log loaded config properties on INFO level
[ FLINK - 13625 ] [ Tests ] Update StatefulJobSavepointMigrationITCase to restore from 1 . 9 savepoint
undid fix for non - issue in checkcast / instanceof XIR snippets
Fix CameraX crash .
Automated [ ] rollback of [ ] .
Use boxed primitives in facade
SPDY : add support for pushed resources in SpdyHttpDecoder
Flip the flag " - - incompatible_disallow_data_transition "
Add show / hideScrubber to DefaultTimeBar
fixed format
Update Bar . java
Solve the problem when project ' s default charset is not utf - 8 ( )
Better diagnosis message for AbstractSocketSslEchoTest
[ JENKINS - 61308 ] Recommend more API like verb / URL to delete items
use an unmodifiable map with a copy of the license request headers
Now understand suffixes in verticle names when deploying
Show overflow button only when there is no enough space
upgrade glide to 4 . 9 . 0
[ FLINK - 19972 ] [ serialization ] add more hints in case of incompatbilities
Ensure withBasicAuth keeps error handler
Register reflectively accessed fields as accessed in analysis
Truffle : added support for Runtime # getCallTargets ( )
[ FLINK - 1201 ] [ gelly ] mapVertices allows changing the value type
Fix bug when fetching unavailable content
Use TableDataSourceNamesMapper to refactor ShardingDDLStatementValidator ( )
Support Fragments which are not hosted by Activities .
renamed ' isJava6VisibilityBridgeMethodPair ' to ' isVisibilityBridgeMethodPair ' ( SPR - 8660 )
Monitoring : Fix MarvelTemplateUtilsTests . java on Windows
Attribute selector : show only ROWID pseudo attributes
Default to visible in GifDrawable .
Don ' t create identity change methods for brand new contacts .
Apply Content - Range and Content - Length headers to resumable upload request .
Update pipeline code to use copyMetaDataFrom method
Fix partial intrinsics bci during bytecode parsing .
Fix optionalLookup to handle substituted types
Refine additional metadata detection logic
Bump version to 5 . 0 . 0 - alpha5 .
am c0ef976e : am 14113e91 : Revert TabLayoutOnPageChangeListener change
moving version to 2 . 0 . 0 - SNAPSHOT
Common DataWithMediaType class and common synchronization for ResponseBodyEmitter / SseEmitter
Cap AudioTrack latencies at 10 seconds and log a warning if too large .
Sulong : Remove use of ByteBuffer in LLVM80BitFloat
JasonLeyba ( on behalf of DouniaBerrada ) : Moving R . java , which is generated by the AndroidSDK , into its own directory .
Ignore cancellation error when search is cancelled ( )
[ ML ] Add ILM policy to ml stats indices ( )
Fix lookup of jvmci and compiler sources im JDK src . zip
Remove unnecessary call to MessageDigest . reset
[ GR - 24053 ] GraalVM LSP : Exception thrown from shutdown ( ) call .
Fix autotest bug
Fix for " does not fit " errors in decompiler " Retype " actions
[ GR - 26445 ] Provide variables if Scope is not the RootNode .
security : cache negative lookups for native roles
Fixed sprite and tiled drawables changing the sprite color .
Support any HttpEntity implementing ResolvableTypeProvider
Add missing allowExperimentalOptions ( ) in tests
dbeaver / dbeaver Disable global breakpoints and add logging of new added breakpoints
make more strict
[ RPBB + aapt2 ] Re - escape XML characters when converting protobuf format back to XML .
[ FLINK - 10102 ] [ docs ] Fix docs for EXECUTION_FAILOVER_STRATEGY
Expose cancelTouchFocus method for convenience .
Clean up unused instance field in SchedulePhase .
Fix package tangle in CLI
Polishing
Add support for alpha versions
Strengthening the reverse proxy admin monitor to detect nontransparent slash encoding ( / vs . % 2F ) .
Fix compilation problem
, alter ColumnDefinition41Packet to RowDescription
[ hotfix ] [ yarn ] Rename YarnTaskExecutorRunner # run to # runTaskManagerSecurely
World : : destroyBody returns fixtures to pool
Truffle - DSL : fixed incorrect else guard connections for executeAndSpecialize .
Moved misplaced assert .
made constant field final
BAEL - 965 - CollectionUtil
Style fix .
Truffle : make Node # replace accept any CharSequence as reason
KAFKA - 6274 : Use topic plus dash as prefix of auto - generated state store names
Unified conversion to JavaKind
Move static method to top of file .
Register JndiPropertySource by default in servlet environments
[ java ] Fixing a legacy test after adding new Tracer abstraction layer
[ GR - 10805 ] Globals bugfixes .
Don ' t use ImageSingletons for UnsafeAutomaticSubstitutionProcessor
WavExtractor : Skip to data start position if position reset to 0
changed testcase .
Office export : cells formatting
Synchronize request init in AbstractHttpSockJsSession
[ FLINK - 5969 ] Also snapshot legacy state in operator test harness
Fix loop redirect error while " CLUSTER FAILOVER "
refactor parseSelectList 14th version
new PEAInliningHints option , turn new options off by default
Clarify what exception gets propagated
Fixed a bug where DefaultChannelGroupFuture is never completed when the specified future map is empty
Clear all caches after testing parent breaker
Tests : move RecoverAfterNodesTests to org . elasticsearch . gateway . local and increase BLOCK_WAIT_TIMEOUT to 10s
Query refactoring : ConstantScoreQueryBuilder and Parser
Improve example project
[ JENKINS - 51584 ] Unit test was not quite correct .
Refactor single result set test cases .
Give option to use a more recent buildtools / api level if found
modified to handle in - memory data source as well .
Update Settings api : fix handling of IndicesOptions parameters in REST layer
[ FLINK - 18537 ] [ table ] Compare only field serializers in RowDataSerializer
[ streaming ] set / get methods written for some basic types
Resources fixes
Transport Client : When adding an address was already added , ignore it , closes .
rename beginPosition to startIndex
Encode MobileInstallCommand failures with FailureDetails
Switch default query to ' select 1 '
modify getEncryptColumnValues ( ) to public
remove erroneous @ Fluent and use warning for now on bad testComplete
fix compile test issue
Enrich store should only update the policies via an update task . ( )
File utils fix
Javadoc .
SimonStewart : Working around a weird hanging bug in the Firefox driver on Windows .
am 009ff4b8 : am 6f800ea6 : Merge " FloatingActionButton fixes " into mnc - ub - dev
Do not use reflection in the tests .
Silence CLI dependency resolution progress reporting when run with - q
improved coverage for package of preprocessor ( )
Validate callback is always invoked in DMLC # stop
Tell NioTask the cause of unregistration
thread dump is added to slave .
Disable Jetty ' s default Server header
improve tests execution time
[ TEST ] Unify the randomization logic for number of shards and replicas
Fixing db indexing integration tests ( )
Improve Http2Test # testResetClientRequestNotYetSent to avoid occasional race
Fix rounding - off error on progress calculation with time - remapped layer ( )
DanielWagnerHall : Use effective , not current platforms , for inline test ignores
RepositoryResolvedEvent : only record the explicitly set original attributes
[ tinker ] Replace ClassLoader in Resources instance .
An UnsatisfiedLinkError from StandardOutputSwapper . ChannelSwapper . swap need not be reported
remove DynamicNewInstance from MethodHandle Intrinsification
Simplify BadgeDrawable ' s constructor to not require anchor view or custom parent view .
Fix Execute and Invoke arity in JavaInterop
Add test for NodeSourcePosition in substitutions
[ tests ] Increase robustness of SimpleRecoveryITCase
Update handled schemes for timing element resolution .
Fix createUnwind .
modify the cross analyzer
TRegex : code cleanup
Disable Failing Debugger Tests
Related to . Missing change .
test : allow percolate api to fail when the percolator field can ' t be found
Perform best effort to retrieve DataSourceProxy
Backported the harden synced - flush PR to v6 . 3 . 0
Don ' t check channels for E - AC3 JOC passthrough
Add ILM policy PUT and GET for remote_monitoring_agent built - in role ( )
Enforce cacheLimit in DefaultSubscriptionRegistry
Update AndroidConfiguration to reflect changes
Randomly use a cached filter .
Support set post sort .
nice usage of versioning parse element
[ GR - 26326 ] Fix race condition in type reachability registration .
Sulong : add allowExperimentalOptions for managed testing suite
Enhance log and null check for rule managers
Returning version injection from File to FileState .
for , fix ms cast exception
[ FLINK - 14673 ] [ hive ] Shouldn ' t expect HMS client to throw NoSuchObjectException for non - existing function
using scan command in delete by pattern
Polish client support for HTTP PATCH
Remove duplicated code in AcceptorChannel classes .
[ GR - 10919 ] [ GR - 10961 ] Sampler improvements .
Automated rollback of commit 5c1005c0947b010ee36ca851b8ba07c9479cf682 .
Fixed bug with scrolling Sprite textures .
Remove unused imports and clean up comment style
KAFKA - 8884 : class cast exception improvement ( )
test improvements
Blacklist Moto Z from using secure DummySurface .
[ hotfix ] [ runtime ] Simplify SlotPoolImpl # maybeRemapOrphanedAllocation
Pass ExtendedEventHandler into MemoizingEvaluator # noteEvaluationsAtSameVersionMayBeFinished .
Optimize Source # read for files and remove double buffering for other input streams .
Polish
Do not attempt to restart non - restartable jobs
[ hotfix ] Additional tests for HadoopDataInputStream # skip vs # seek
Test for disconnected client detection
If the cluster does not contain any indices then index templates do not survive cluster restart , closes .
Post preliminary events so it doesn ' t crash outright .
Fixed copying TintedDrawable .
Rename ChannelHandlerContext . type ( ) to ChannelHandlerContext . types ( ) as it return a Set
DB2 : Display the exception in the logs when DBEaver can not " load " schema SYSIBM
[ GR - 19539 ] [ GR - 19540 ] Fix handling of - H : + ExitAfterAnalysis and - H : + ExitAfterCAPCache .
Upgrade to Spring Doc Resources 0 . 2 . 2 . RELEASE
Update stroke instead of creating background again
ArgumentListBuilder2Test . slaveMask failure .
Moved a bunch of lottie parsing into RootLayer
Record uncompression scale in AMD64AddressNodes
fix : remove declaration of unused variable ( PR )
[ GR - 13781 ] Remove the deprecated Remote option .
[ GR - 23513 ] Support license URLs without embedded license files .
only record field accesses if GeneratePIC is true
[ JENKINS - 16332 ] Trying to determine if CanonicalIdResolver is implicated .
Fix file handle leak in readBlob method of AbstractFsBlobContainer
allow callers to specify their own Handler
[ FIXED JENKINS - 4220 ] fixed a dead lock .
StandardOp . NoOp : assert that the right instruction is # replaced .
Better error message .
[ FLINK - 13216 ] [ FLINK - 13153 ] [ table - planner - blink ] Fix Max_Retract and Min_Retract may produce incorrect result
Resolve TODO now that the feature configuration supports conditional expansion
Fix
Integrate reviews and add javadoc .
[ ML - Dataframe ] Use standard string constant for mapping type ( )
Strictly enforce reverse deps removals .
JNIFunctions . RegisterNatives should throw NoSuchMethodError
Remove the need for generated accessor methods .
[ ML ] Index the audit notification before responding to a delete request ( elastic / x - pack - elasticsearch )
[ FLINK - 8161 ] [ tests ] Harden YARNSessionCapacitySchedulerITCase
Demo app : Do not allow tunneling on API < 21
Remove warnings from RingBufferEventMatcher .
Skip ads before the initial player position
sulong : fix EagerParsingLazyFailingTest
Fix @ Alias fields
Cache computation of timer and metric objects for phases
Fix NoSuchElementException when BuildEventArtifactUploaderMap has no elements .
[ JENKINS - 30139 ] Oleg wants to keep the data accessible so we have to go for immutable
bug fixed for asm serialize
only retrieve transaction manager bean if necessary ( SPR - 7153 )
Remove flag control of turbine bridge pruning
[ FLINK - 4569 ] [ tests ] Respect exceptions thrown in thread in JobRetrievalITCase
Remove isolated use of Reactor Buffer
[ bs - 62 ] Add assertions to CLI integration tests
[ hotfix ] [ tests ] Refactor unit tests in RemoteInputChannelTest to avoid mock way
Implementing Code Review comments - 3
Report dependency version we failed to load , not the one we require
Add an HTML5 encoder mode for HttpPostRequestEncoder
Fix EndpointRequest ignore logic
[ GR - 7818 ] Unknown signal POLL .
Await for close of the channels . See
[ FLINK - 12673 ] [ network ] Introduce NetworkEnvironment . getUnreleasedPartitions instead of using getResultPartitionManager
[ hotfix ] [ core ] Drop hijacking of TypeHint by OutputTag . Improve error message for OutputTag .
Fix number conversion tests .
bug fixed for ValueFilter
Add missing tests for actions whose inputs do not change
Mark locks without MonitorIdNode as eliminated
Add CC_FLAGS to the macros that pull in the CcToolchainProvider for genrules
[ fixed ] Texture can now load ETC1 textures directly .
refactor ExecuteUpdateResponse
Make docker nodes acknowledge they can run sessions
Fix incorrect usage of TruffleRuntimeOptions . getValue ( ) on PolyglotCompilerOptions
BottomNavigationView : contentDescription updates
Enable SSL for blob storage
[ Mike ] Modified performance testing framework to do more runs and print information as it runs .
Fix insertion of profile data in unit tests .
[ FLINK - 1039 ] Fix pojo expression keys for group reduce
[ FLINK - 19291 ] [ avro ] Fix exception for AvroSchemaConverter # convertToSchema when RowType contains multiple row fields
Xml resultMap Error message location
use stream API to concatenate ZipFile iterators
Fixed suspected NPE .
remove double check for null in value source
[ hotfix ] [ checkpoint ] Remove NonNull annotations and remove trivial doc in OperatorSubtaskState .
REST API : Allows all options for expand_wildcards parameter
Use ConcurrentHashMap for ConstantFieldInfo cache .
Update ActiveMQConsumerInterceptor . java ( )
KAFKA - 5068 ; Optionally print out metrics after running the perf tests
LSRA : findOptimalSplitPos : skip block iteration when loop depth is 0 .
[ FLINK - 11485 ] [ core ] Refactor multiple nested serializer compatibility resolution logic to CompositeTypeSerializerUtil
treat 64 - bit and 32 - bit sized global types interchangeably when moving to native , and fail for other types
Fixing noProxy handling in Grid
KAFKA - 9165 : Fix jersey warnings in Trogdor ( )
[ FLINK - 3570 ] [ runtime ] Use InetAddress . getLocalHost ( ) as heuristic to find local address
polishing
Cassandra data model
improved sql parser .
Fix ClassCastException bug when old incompatible option value
Fix typos in Context . Builder
Backport the HttpPostRequestEncoder fixes by @ igstan
improve sql wall
BugFix : WS_DOMAIN_NAME , SUBGROUP default values override custom values passed by java options options
NPE fix
Devirtualizing interface call should respect the maySpeculate flag .
RenderOnDemandTest compiling against HtmlUnit 2 . 17
refactored a AES - 128 key generation function
Introducing the launchOptions with more elegant way
RList . getAsync method added
- throw InterruptedException from CacheLoader methods
For - - aspects parameter , ignore top - level targets that are not rules .
Fix additional deprecations
Disable heuristic_label_expansion in Bazel .
Fix test failure
Tabbed editor render fix
TraceRA : add reorderTraces ( deactivated ) .
adding network connectivity check first
Remove some occurrences of gendered pronouns
[ FLINK - 13250 ] [ blink runner ] Make sure that all nodes have a concrete resource profile
C3P0Adapter
Fixed - NPE in CommandPubSubDecoder .
everything i do , i do it for you
Fix proto documentation typo
AdvancedButtonTest compiling against HtmlUnit 2 . 17
Handle laggy manifest servers . Don ' t reprocess discarded manifests twice .
[ SECURITY - 54 ] Massaging the original fix .
[ FLINK - 12612 ] [ coordination ] Maintain JM connection until all partitions are released
Have a dedicated join timeout that is higher than ping . timeout for node join
Adding PerformanceInformationHandler to the logging white list .
Image copying : check if total = = 0 , use default total size
[ hotfix ] Allow leader assignment to TestingLeaderElectionService if it has not been started
Do not set timeout for IndexRequests in GatewayIndexStateIT ( )
Unmuted testMonitorClusterHealth test and added more logging . ( )
[ perf ] TraceRA : remove FastSSIBuilder . BlockData
[ GR - 19116 ] Be tolerant to invalid isAnonymousClass implementations .
GT - 2974 PDB Analyzer using CategoryPath forward slash capability
SQL editor fix ( ot work prior of 4 . 2 . 1 )
Support configuration of Flyway ' s Pro properties
for , add DatabaseFrontendEngine . createPacketPayload
[ tinker ] Use PathClassLoader to trigger dex2oat after patch is applied .
Prevent BitmapFactory from overriding mark size
Allow MP3 files to play with size greater than 2GB .
Fixing Test Cases
Make removal of non - existent cache span a no - op
The lock used during executing a fired alert should also encapsulate the updating of the fired alert .
PG : views and procedures rename support
OpenSslEngine encrypt more data per wrap call
Report progress during query command
determine if an Object is of primitive type
Promote Observable to Single / Flowable adapters to top - level types . ( )
Replace one use of String . split ( ) with a Splitter to reduce
Fix flakiness in RealWebSocketTest . clientAndServerCloseClosesConnection
GT - 3323 - GTree - review fixes
Composite cache load fix ( skip null objects )
Fix NPE in InboundHttp2ToHttpAdapter
Reliably refresh metadata for dynamically changing prototype bean class
Upgrade javaparser ( )
Use runOnContext to run waiting requests to avoid stackoverflow
Generate header file for dynamically loaded SVM images
Dynamic FindReplaceTarget in Data editor
Incremental search fix ( check ranges )
[ FLINK - 11232 ] [ rest ] Fix subtask start - time field name
Add support for voice selection in WebVTT CSS
dump final HIR schedule to c1visualizer
fix segments per tier to the proper 10 default
[ hotfix ] Minor clean - up in TaskExecutorProcessUtils .
Fix ScheduledJobsIT failure
Minimal change to expose segment indices in DefaultDashChunkSource
Move exception from PostAnalysisQueryBuildTool to BuildTool .
Test the Gradle Plugin against Gradle 4 . 10 . 2
local message bucket
Fix engine benchmark should not need polyglot access .
SQL formatter ( delimiters , SET , BEGIN / END )
bufix for json array
Extracted FreshValueGenerator from ForwardingWrapperTester to a stand - alone class so it can be reused for automated equals testing .
[ GR - 4098 ] SignedDivNode : Do not omit fs when doing a canonicalization .
Pass canonicalizer into IterativeConditionalEliminationPhase .
update SpringNamespaceExample
Project md read error handler
Searchable snapshot rolling restart ( )
Add RouteSQLBuilderTest
TraceRA : partly reverting .
Using FrameLibrary to access Frame values effectively
Improve efficiency of reverse queries for Skyframe by memoizing getAllowedDeps calls on the same batch call
[ TEST ] Make InternalEngineMergeTests more stable
Stubs hang onto intermediate compiler state forever
bug for long type builder setter
Restart any non - running identical requests instead of ignoring them . ( )
[ fixed ] issue 894 , CompositeAction # finish ( ) produced garbage due to enhanced for loop .
SPDY : replace exception in SpdyHeaderBlockZlibEncoder with EMPTY_BUFFER
try to fix test failures by adding timezone support to PlainTestMessageCodec
Simplify code generation of reinterpret - memory .
SPR - 8698 Support flash attrs and a ModelAndView return value .
KAFKA - 6512 : Discard references to buffers used for compression ( )
Handle illegal NewInstance .
fix neverPartOfCompilation that isnt ' s statically unreachable
Refactor JDBCExecutorWrapper
[ grid ] Fixing DistributedCdpTest
Reinstate commented - out CoordinatorTest ( )
[ Engine ] checkVersionMapRefresh shouldn ' t use indexWriter . getConfig ( ) . We run it out of lock , the indexWriter may be closed . .
Cleaned up sort - merger test .
Disk decider respect watermarks for single data node ( )
Update HttpHeaderDateFormat . java
remove casting
SslHandler wrap memory leak
Clean up WebvttParser somewhat .
Remove defaults from Apple device types and versions
Test the Gradle Plugin against Gradle 6 . 0 . 1
Move OS - independent code into JavaThreads
remove ( R ) collides with remove ( Object ) at AbstractLazyLoadRunMap . So renaming a method .
BAEL - 1107 : update formating and version of libs
java : Adding two more cookie tests
Prevent endpoint ' s shutdown ( ) method from being a destroy method
Add a better error when v3 and v4 modules are used in the same app .
[ hotfix ] [ streaming ] Initialize StreamingRuntimeContext to rich functions early
Changed Basic Authentication consumer to setBasicAuth method
[ hotfix ] [ datastream ] Fix the formatting of StreamEdge class
GT - 3455 - Update action key binding collision dialog to allow for mouse
Reworked the fix .
[ feature ] tear down Truffle compiler threads
Project export fix
Added bulk copy methods to memory segment .
[ hotfix ] [ streaming ] Refactor TwoPhaseCommitSinkFunctionTest
[ tinker ] bugfix : NPE was thrown if new apk has more dexes than old one .
Allow 0 ( use default ) for backlog value
Do not throw / catch exceptions during engine data lookup .
[ hotfix ] [ connector - kafka ] Merge KafkaITCase and KafkaTableTestBase
MySQL quote formatting fixed
modify the uploader
SPR - 5536 : RestTemplate does not do HTTP GET if it should
Polish
HOTFIX : Fix wrong setting of Serde in ` MeteredTimestampWindowStore ` ( )
[ FLINK - 11835 ] [ tests ] Wait until job was recovered before unblocking task
Minor changes for SortedGrouping
Find SourceSection in Wrapper Nodes
Instead of using the SortedLists utility class , go straight to Collections . binarySearch in the implementation of RegularImmutableSortedSet .
Use ExoPlayer ' s AnalyticsListener and AnalyticsCollector .
Configures HTTP2 pipeline with more proper way
[ grid ] Using Duration . ofSeconds instead of ChronoUnit [ skip ci ]
Make PerThreadIDAndVersionLookup per - segment
Ensure that CLI - based tests automatically import new TestRestTemplate
Issue number : 747
Mapping - Date Format : 3 or more custom date formats fail using ' | | ' delimiter , closes .
Changed test to meet refactored class
[ changed ] version at 0 . 9 . 6 now : D
Store genquery ' s scope of accessible labels as a Label - > Target map .
Fixes 5103 . Items under a ScrollPane ' s fading scrollbar are not clickable even if the scrollbar is hidden ( ) ( )
[ JENKINS - 38711 ] Add UncaughtExceptionHandler to remoting related threads ( )
loop peeling : cleanups
[ TEST ] make BulkTest more robust if test infra is slow
Minor improve
LSStackSlotAllocator : use priority queue .
Pass - - keep - main - dex flag to rex when multidex mode is set to legacy
TEST : Use non - zero number for # testCompareUnits
Add executable argument to repository_ctx . download function .
Sulong : caching truffle file for context sharing
[ BAEL - 14274 ] - Fixed article code for https : / / www . baeldung . com / mockito - annotations
Fix a class loader leak in ForkJoinPool
Starlark : inline the method Dict . putAllUnsafe , update the comment
modify nullpoint exception
Core : Use reading / writing search request infra also in alert action
RS filtering fix ( keep original RS metadata after adding filter )
Change DefaultChannelId visibility to default . Related to [ ]
Mapping API : Improve IP address validation
Use system variables in driver files path
modify the position of parseConfigMap .
Binary Graphs : use 16bits for pool indices
Public RoundedDrawable
Setup cache in a loop .
, refactor ExecuteQueryResponse
SimonStewart : Fixing the test_selenium target so that it runs
Javadocs .
Fix virtualization of double word Unsafe operations overwriting double word value
[ GR - 23076 ] Fixed review comments .
Only nudge NinePatch UVs if the texture uses linear filtering .
Do not use two volatiles to hold state that needs to be atomically updated
Fix typo for DOM . attributeModified
Color fix
Reduce memory copies in HttpContentDecoder and so also the risk of memory leaks
am 193f33fa : am 40fbb22c : am 2521e5cf : am 0b004890 : Fix FAB not returning after Snackbar dismissal
adapt to upstream changes
[ FLINK - 15347 ] Bump AkkRpcVersion to 2
Add cc transitive headers to ide - resolve output group
Properly propagate errors that occur during preparation .
Pass the ringBuffer to EventProcessorFactory . createEventProcessor instead of making each instance fetch it from the disruptor .
[ added ] applet audio recorder , untested , applet needs permissions probably . . .
[ JENKINS - 14923 ] Removed CONSISTENT_HASH flag to simplify code .
Moving utility method isScriptFile to the only class where it is needed
fixing recovery debug logging param mismatch
Extract CONTAINS_JAVA_PROVIDER to a shared constant
let prelert metadata test extend from AbstractSerializingTestCase , so that we test the writable and xcontent serialization much better .
Fixed drawing full circle without slice spacing
MINOR : Remove ` PartitionHeader ` abstraction from ` FetchResponse ` schema ( )
SNAPSHOTS + TESTS : Correctly Wait for Clean State ( )
Query performance : ForkJoinPool for LabelVisitor
KAFKA - 5636 : Improve handling of " early " records in sliding windows ( )
Added substring methods to PactString
added another overload that allows the caller more fine - tuning .
OkHttpClient . cancel ( tag ) doesn ' t cancel in - flight synchronous requests .
[ FIXED JENKINS - 22767 ] Make sure only one thread actually loads a given build .
DanielWagnerHall : Fixing ignored test printer to work with junit 4
fix : Sort Processor does not have proper behavior with targetField ( )
[ FLINK - 10399 ] Rely on exception propgation to fail tests .
Use test executor rather than mock in ExtraActionTest .
experimental UI : also during loading produce a short progress bar
Fix broken tests
Remove method - based IC dispatch .
Update SENTINEL_VERSION constant to 1 . 6 . 1
turned TypeDescriptor . UNKNOWN into an internal detail ( SPR - 7349 )
[ fixed ] box2d character controller , need to reset friction in latest box2d version .
Removing IntersectionOrder . SENTINEL and adding Edge . MAX_COORDINATE for code readability .
Add nullness annotations on the fields in CompactLinkedHashSet . java
nul = = xxx
Bump version in BWC check after backport
Uses Charset instead of String in MimeType . equals ( )
Consolidate one more calendar method . Insert events with ACTION_INSERT not ACTION_EDIT
EmptyObserver
Improve performance of DispatcherServletRegistrationCondition
Make Ouya mappings constant expressions
DanielWagnerHall : Don ' t use deprecated HtmlUnit methods ( issue 891 )
needs to add loaded Fingerprint to cache .
query type fix
Jackson2ObjectMapperBuilder allows for regular constructor usage as well
Only skip dexes which is not belong to our app to support enhancement frameworks .
Implement authentication
SPR - 8453 - SpringOXM CastorMarshaller tests
Fix ValueLocation . set .
JSQLParser version update . Conditions editor .
GT - 3343 - File Browser - review fix
JavaInterop : cleanup and minor fixes in message implementations .
Add PersistentTasksClusterService : : unassignPersistentTask method ( )
[ changed ] GdxNativesLoader , commented out error message .
Fixes deadlocks in LocalCache . asMap ( ) . compute ( ) if the Function throws
Test : Simplify watch stats test script engine ( elastic / x - pack - elasticsearch )
Fixing key of pageLoad timeout
Added another integration test to check Nephele ' s fault tolerance capabilities
Why are these binaries tracked , Mario ?
HOTFIX : fix flaky StateDirectoryTest . shouldReturnEmptyArrayIfListFilesReturnsNull ( )
modify JavaDoc and remove the default value .
Break cycle caused by JndiDataSourceAutoConfiguration
[ hotfix ] [ tests ] RocksDBAsyncSnapshotTest : use test harness helper method to wait until task is running
Mock data : error messages pring fix
Value handler update fix
Use LinkedHashmap to preserve insert order
modify the cat
Feature : compatible with acm ( )
[ Dubbo - 4871 ] fix cache file name issue in windows platform ( )
improved testcase
PG : add TABLESPACE keyword
[ removed ] redundnat null glBufferData . silly me
[ FLINK - 1514 ] [ gelly ] Simplified GatherUdf ; added forwarded fields annotations
Fixed argument check
[ FLINK - 12983 ] [ metrics ] replace descriptive histogram ' s storage back - end
x - content : Support collapsed named objects ( )
Add timing logs for runners .
Save As for SQL editor
KAFKA - 9457 ; Fix flaky test org . apache . kafka . common . network . SelectorTest . testGracefulClose ( )
annotate generated Node classes with @ GeneratedNode ; improved error messages in @ NodeInfo annotation processor
Corrected typo in checkpointing annotations
disable PI tests
Hibernate synchronization properly unbinds Session even in case of afterCompletion exception ( SPR - 8757 )
Fix extra bytes and missing resources for roboelectric tests :
GraphBuilderConfiguration . copy ( ) should do what it ' s javadoc says
Remove the - - artifacts flag from the dump command , which was long deprecated .
[ hotfix ] Cleanups of the AbstractKeyedStateBackend
Switches to micrometer metric name
HOTFIX : Fix HerderRequest . compareTo ( )
SimonStewart : Extracting some of the logic for handling extensions from the firefox profile class into it ' s own home .
Tests - updated test for options rename
Added a flushStatments after a query to fix issue http : / / code . google . com / p / mybatis / issues / detail ? id = 138
guard against npes
code cleanup : remove unnecessary and optimize the function ' s parameters . ( )
Fixed - RRateLimiter . availablePermits ( ) method throws exception ( regression since 3 . 13 . 3 ) .
Removed forcedDateParser from hashCode ( ) and equals ( ) as it holds no value when it comes to equality .
security : allow indices monitor actions to access the security index
revised use of id and display name ( id may be null ; SPR - 5761 )
MINOR : Replace empty string concattenation with String . valueOf ( ) where possible . ( )
Refine the behavior of detecting unresolvable types .
Track total hits in tests that index more than 10 , 000 docs
[ ML ] Remove excessive logging from ML upgrade mode ( )
extract 2 methods : ( )
randomDir name fixed
Cleanup format in LinearInterpoatingScorer . java
[ GR - 14180 ] Replace usages of File . separatorChar with Truffle API .
db2 sql parser support hierarchical .
DashMediaSource cleanup
Set the new lucene version for 6 . 4 . 0
[ ML ] Create state index and alias before starting an analytics job ( )
use dataSource . getShardingContext ( )
SQL : Extend the multi dot field notation extraction to lists of values ( )
[ FLINK - 9692 ] [ kinesis ] Harmonize style of config variable names
modify the cat
Automatically add experimental / deprecation documentation to guarded Params
use execute instead of executeQuery in getTableNamesFromDefaultDataSource
[ GR - 7017 ] Allow for fewer occurrences of retrying message .
expose User . XSTREAM2 so that plugin can register backward compatibility aliasses
Use Paths . get insted of Path . of
Fixed a problem where HttpTunnelingClientSocketChannel . setInterestOps ( ) returns a wrong future
[ GR - 25386 ] Fall back to copying the image heap when it cannot be found / opened .
Set subSearchContext . topDocs after the rescoring in TopDocsAggs
compress cluster state when publishing it
DefaultLifecycleProcessor properly counts dependent beans in same phase
re - enabled disabled test
CR - 1364
Minor improvement to NetTest StartTLS
Fix StorageSyncHelperTest .
[ GR - 24622 ] VMInspection . isEnabled ( ) is assumed to return the AllowVMInspection value .
Filtering for nested ERROR dispatch
[ hotfix ] [ tests ] Let MarshallingTestBases read the value from String value
HttpClient does not use HTTP proxy when client options ssl = true overridden by request options ssl = false - fixes
Add type information to thread - local printing .
Preserve selection when setting new items , if possible .
Apply DataSource aliases only when necessary
[ JENKINS - 19082 ] Fix ListView includeRegex
clean code ( )
Allow TruffleOptions to be set using system properties .
remove heartbeat graph
Address issue 952 on public guava : don ' t make spaces in parameters mandatory .
check whether a StateSplit node being added via GraphBuilderContext has a null stateAfter before setting one
SSH password prompt fixed . Datasource config change listener fixed .
Produces media types cleared prior to error handling
Start using return value of OnNavigationItemSelectedListener . am : 578015fc3e am : a502cd075f am : ba4b4c48c7
[ FLINK - 12204 ] [ jdbc ] Improve JDBCOutputFormat ClassCastException .
class might not be public even when a method can be .
core : fix type check for loop over iterable .
Don ' t drop metadata annotation in ConstantNode smart constructors .
Strengthen handling of paths with spaces .
compilation failure in jdk 5 - > use Spring solution ( return Logger . GLOBAL_LOGGER )
Emit appropriate LoadCompressedObjectConstOp mov
remove unused code
Rename ModuleEnvironment . getFileFromDepot to getFileFromWorkspace .
MySQL column charset / collation edit
update ILM integ test cluster poll interval to 1s ( )
Polish MimeType
Fix ColorDrawables not being rendered properly on Lollipop ; closes
Fix issue where quoted replies showed a ' save ' option .
[ ML ] Fix deleteCategorizerState ( ) by using delete by query instead of individual blocking delete requests .
Update tests to use AnimationController
Upgrade to Couchbase Client 3 . 0 . 6
[ TEST ] Ensure ` InvalidateApiKeyRequest ` has at least one key in test ( )
Took care of caption styling priorities
[ FIXED JENKINS - 13625 ]
TraceRA : improve assertion in TraceBuilderResult .
Add StrikethroughSpan support to SpannedToHtmlConverter
use getDefaultKeyGeneratorConfig ( )
WebClient supports Void response body
REPL Debugger : command to set language now case - insensitive
Fix re - starting of DefaultHlsPlalyistTracker
KAFKA - 8265 : Fix override config name to match KIP - 458 . ( )
WriteBarrierSnipppets : make locations visible to other packages
[ GR - 13311 ] TruffleFile doesn ' t work in the root directory .
[ GR - 16963 ] Add support for SecurityManager . getClassContext ( ) .
sandbox : Create more compact sandbox profiles faster .
Can be inlined must also return true for native methods that are intrinsified .
Stop responding to ping requests before master abdication ( )
JiayaoYu : Make WebElement . findElementByName / Id search recurviely descending ( Firefox and HtmlUnit driver )
[ Transform ] fix time rounding in TransformContinuousIT ( )
Clear mask canvases on pre - Pie ( )
Updated HtmlUnit BrowserVersions to FIREFOX_38
Remove unused constructor arg from UserDestinationMH
[ ] Make sure HttpContentEncoder calls ByteBuf . retain ( ) if needed
Defer injection of top - level context until we are just about to do the execution phase .
Quick workaround for buffer deadlock problem
Polish DatabaseStartupValidator
[ JENKINS - 61521 ] Ignore RemoteClassLoader # fetch3 on telemetry
Fix copy_to when the target is a dynamic object field .
RBatch errors handling
[ hotfix ] [ rest ] [ tests ] Fix Preconditions placeholder
fix parsing mysql 8 create table sql with check not enforce that can ' t recognize the " NOT " token
Prepare to abstract away the param file write action from tests .
print more details when ProcessImpl error
Aggregations Refactor : Refactor Cumulative Sum Aggregation
Remove code that allows RunfilesSupport . owningExecutable to be null .
Add a new ConfiguredTargetKey subclass for configured targets with explicit toolchain contexts .
blocking synchronous next
Fix : ' TO_NATIVE ' MR should wrap return value if different to receiver .
Handle the backslash with double quote in JsonObjectDecoder
Adjust the versions for ' fields ' in inner_hits .
Separate Source - Level and Java Debugging of LLVMPointerImpl
Fix incorrect usages of org . mockito . internal
Introduce Label . EXTERNAL_PACKAGE_FILE_NAME as a convenience . Note that we already have Label . EXTERNAL_PACKAGE_NAME .
Additional check that key can be acquired ( issue )
SQL Developer parse json
[ FLINK - 6182 ] Fix possible NPE in SourceStreamTask
[ hotfix ] [ tests ] Fix Files . walk resource leak
Improve precision of Location tests
[ hotfix ] [ network ] Release unpooled buffer for events .
Remove erroneous ' + ' from log message .
Fixed GRAAL - 276 .
SPR - 8453 - SpringOXM CastorMarshaller tests
PubSub improvements
Generalized workaround for JDBC 4 . 1 getObject ( index , requiredType ) failures on MySQL and Derby
WordCount examples are documented with many comments in order to be very user - friendly .
Corrected Help Location for Find Checkouts Dialog
Implemented algorithm to compute connectionIDs
[ GR - 23142 ] Sulong : add lli toolchain api .
Actor , fixed leaked clip Rectangle .
SQL delimiter handle fix ( SQL Server )
Enable starlark profiling by default again .
Proposals sorting
SimonStewart : Implementing getTable in the webdriver backed selenium class
Remove un - necessary incorrect synchronization
Relax ExoHostedTest # MAX_PLAYING_TIME_DISCREPANCY_MS
test
Fix signature of createComponents after addition of script service in core
Use Gradle ' s CreateStartScripts task with 6 . 4 and later
improve logger output on startup
SimonStewart : Deal with partially loaded pages not executing JS properly when waiting . Address issue 1972 .
Tweak MMS retrieval and submission .
[ Storm - Compatibility ] Forward Storm Kryo registrations to Flink
[ FLINK - 2936 ] Fix ClassCastException for Event - Time source
[ AArch64 ] Make AArch64IntegerArithmeticSnippets work .
Add GWT src - super for HashBiMap change .
Expose AndroidPreDexJarProvider to Skylark as AndroidPreDexJarInfo .
[ FLINK - 9552 ] [ iterations ] fix not syncing on checkpoint lock before emitting records
[ GR - 17729 ] Remove usage of SVM unsupported SecureRandom .
Add a ` Capabilities . merge ` method
Clean up unnecessary use of full package name
Fix empty strings in empty lists in cc_configure
[ FLINK - 16222 ] [ metrics ] Support loading reporters as plugins
Improve HTTP message streaming in decoder , in Master same as submitted fix
Apply small improvements in the inline assembly parser
for , remove ShardingInsertColumns . insertStatement
run netx without requiring X
for , RuleDefinitionFileConstant . FILLER_DEFINITION_FILE_NAME = > FILLER_RULE_DEFINITION_FILE_NAME
[ hotfix ] [ tests ] Make JarHandlers re - usable
Integrated json - lib 2 . 4
minor metric fix
[ java ] Using try - with - resources to close the stream
Making " beta " console the default one , it ' s time to get out of beta status
Test : Don ' t force networking mode .
[ ML ] Ensure class is represented when its cardinality is low ( )
Adjust version on final pipeline serialization
[ hotfix ] [ runtime , tests ] Fix checkstyle violations in ThrowableClassifierTest
Sync Reactor SignalEmitter
Refactored failure patterns editor
Implemented queue for pending actions inside the task manager component of the Nephele streaming plugin
Use D8 package private bazel specific APIs
Load Media Variations sample images after layout
KAFKA - 6464 : Fix Base64URL encode padding issue under JRE 1 . 7 ( )
Handle partial reads / skips in ImageHeaderParser .
Re - instated Jetty 7 , this time with a well - placed flushBuffer
fixed HTTP invoker to support resolution of multi - level primitive array classes again
[ FLINK - 14782 ] [ table ] CoreModule # getFunctionDefinition should return empty optional when function does not exist
Implemented support for FieldNamingPolicy with MiniGson .
Adjust scope of Groups V2 feature flag .
modify the bug http api
Minor changes to TearDownStack :
Stop blaze crash due to a symlink cycle or unbounded expansion encountered
[ FLINK - 14415 ] [ table - common ] ValueLiteralExpression # equals should take array value into account ( )
[ FLINK - 13617 ] Update FlinkKafkaProducer011MigrationTest to restore from 1 . 9 savepoint
[ FLINK - 10739 ] [ tests ] Harden ProcessFailureCancelingITCase . testCancelingOnProcessFailure .
Improve watcher smoke test stability
KAFKA - 3169 ; Limit receive buffer size for SASL packets in broker
[ ML ] [ TEST ] Readjust AutodetectMemoryLimitIT assertions
ellipse off by default .
Reverse order of active profiles before processing
Fix native - image - maven - plugin for Windows
Add SniCompletionEvent which allows to easily retrieve the hostname that was used to select the SslContext .
for , refactor XXXOrPredicateFiller . getTableNameFromMetaData
Make Java integration tests more consistent with the depot
avoid retransform lambda class .
Fix data migration and add a test
Empty SQL script confirmation handle
Add saved state
[ ML ] Add support for percentiles aggregation ( elastic / x - pack - elasticsearch )
BAEL - 2969 : Copying Sets in Java
[ ML ] Fix custom timestamp override with dot - separated fractional seconds ( )
scene2dui scroll bars can now flip left / right and top / bottom , added ScrollPaneScrollBarsTest to tests
Fix flakiness in authenticateWithChunkedStreaming ( ) .
WavExtractor : Skip to data start position if position reset to 0
Refactor SpringApplication Environment config hooks
Fixed a compilation error .
add testcase
MINOR : Use ` Topic : : isInternalTopic ` instead of directly checking ( )
remove redundant code . . .
Fix index computation in ByteArraySequence .
[ GR - 6314 ] Skip deleted nodes in ` registerCombinedStamps `
Add unicode emoji for tests passed .
Control command parsing fixed
Remove object canonicalization mechanism from NativeImageHeap
Deprecate some ViewUtil methods . Inline others . Remove some old API code .
[ tinker ] Bugfix .
Conscrypt 1 . 3 with TLSv1 . 3 ( )
[ FLINK - 20033 ] Make JobManagerJobStatusListener call back run directly
change delimeter for username / domain concatenation
[ FLINK - 10390 ] [ metrics ] [ datadog ] Close responseBody
Counting the number of words in string
Modify EventMessageDecoder to return null if decoding fails ( currently throws exceptions )
less locking
KAFKA - 6101 ; Reconnecting to broker does not exponentially backoff
merge code
shorten tier names
[ FLINK - 7728 ] [ DataStream ] Simplify BufferedValveOutputHandler used in StatusWatermarkValveTest
SparseFileTracker . Gap should keep a reference to the corresponding Range ( )
Print an error message with remote host details when a DNS rebind attack is detected ( GR - 24140 ) .
[ ML ] Fix line length
more checks for EventBusAnnotationProcessor
[ GR - 3892 ] Fix arity deopt loop .
Prevent frozen frames when the decoder is always late .
Removed unnecessary supressed warnings .
[ TEST ] Reduce likelyness of sync on every operation
[ FLINK - 2356 ] Add shutdown hook to CheckpointCoordinator to prevent resource leaks
TransportClient does not load plugins , closes .
AjayKemparaj : Made Grid Help look cleaner , previously it was ugly , i am guilty for code duplication of
request handler methods with @ ModelAttribute annotation always return a model attribute ( SPR - 4867 )
Avoid repeated copies of RuleClass attributes .
Quoted names processing fix
[ Dubbo - 3570 ] repackage compatible enhancement . ( )
MINOR : Trogdor should not assume an agent co - located with the controller ( )
Keep main dex check enabled in D8
Fix test bug : we need at least one parent to have 1 child or more .
Better behavior if media doesn ' t support DRM scheme
RLocalCachedMap . clearLocalCache method added
Mixed key types for join
[ hotfix ] [ network ] extend logging message in SpillableSubpartition
[ java ] FirefoxOptions . asMap should return an immutable map ( and its subelement should be immutable too )
DownloadManagerTest : Clean up the tests themselves
Add server healthy method
add serial execute engine while in transaction .
Allow rules to consume CcLinkParamsProviders provided by Skylark rules .
[ changed ] BufferUtils uses Array for tracking .
Changed use of AssertThrows to @ Test ( expected = . . . )
Avoiding Class . forName when running in ahead - of - time mode
Fixed indefinite blocking
ERD attributes type name fix
Fix NPE when importing markdown files
Clarify error message on keystore write permissions ( )
Fix clearkey DRM UUID passed to MediaCrypto
Test the Gradle Plugin against Gradle 5 . 6 . 1
[ hotfix ] Fix the code style in BashJavaUtilsITCase
[ FLINK - 11086 ] [ e2e ] Properly add Avro JARs in SQLClientKafkaITCase
Fix card ripple bug
Remove some unused code
ProducerName is public for MediaVariationFallbackProducer
These permissions should be RUN_SCRIPTS
Truffle : make Node # replace accept any CharSequence as reason
MINOR : Improve producer test BufferPoolTest # testCloseNotifyWaiters . ( )
DouniaBerrada : Renaming test .
No Content - Disposition if HTML in the request mapping
KAFKA - 8412 : Fix nullpointer exception thrown on flushing before closing producers ( )
[ streaming ] Refactored the NumberOfInputs setting in JobGraphBuilder
Value extraction
add process lock for interpret mode
handle unknown protocols better ( though currently only ftp , http , https makes sense )
fix a bug for send mail
Add back deprecated MediaSource . prepareSource for ExoPlayerSampleExtractor .
[ FLINK - 10075 ] [ rest ] Addnull check before KeepAliveWrite . flush
Clean up testing of constraint - based selects to not rely on anything that uses toolchains .
, fix test case
Add CGlobalDataStrings com . oracle . svm . core . VM . Target . { Platform , LibC , Libraries , StaticLibraries }
Faster and Simpler GCS REST Mock ( )
Workaround ResolvableType . forClassWithGenerics bug
setString ( )
[ hotfix ] Fix checkstyle violations in SlotManager
[ FLINK - 19306 ] [ coordination ] ResourceManager supports declarative resource management
RSV ordering fix
Mappings : Fix _field_names to not have doc values
Making the concurrent jobs to wait longer under travis
- fixed
Introduced sequence numbers for input split requests to handle recovery of input tasks
KAFKA - 3393 ; Updated the docs to reflect the deprecation of block . on . buffer . full and usage of max . block . ms
Text editor context menu fix
Fix test failures due to incorrect validation
Adjust PluginInfo . LICENSED_PLUGINS_SUPPORT value after backport
remove invalid import
ERD : node UI settings save / load
Call onFailure for every exception case even in the case of an error / runtime exception
Fix various period preparation and source info refresh error throwing issues
Convert keyword case fix ( use formatter keywords case )
Fix unassigned value .
Add proper live stream clipping .
Make JDK version check more robust .
Clean up JNIRegistrationJavaNet . duringSetup
[ streaming ] Updated StreamComponentTest
core : fix enum reconstruction ( )
fix , initCommonParseRuleDefinition = > initGeneralParseRuleDefinition , initParseRuleDefinition = > initDialectParseRuleDefinition
Handle legacy DASH audio channel configuration
2 . x : enable operator fusion in onBackpressureBuffer ( )
PostgreSQL triggers
Add service setting
Add libopus / libvpx availability checks
ByteBuffer support postponed
Multiplying strings with negative numbers no longer leads to an exception .
[ added ] ParticleEmitterBox2D .
update version
Segregated test cases for primitive and object types . Added test case
fix for http : / / code . google . com / p / mybatis / issues / detail ? id = 187 ignore buffered statements in case of rollback
fix output encoding problem . close
Turning NodeClass into abstract class to allow its independent implementations .
Avoid using immutable collections for configuration proprerties
Remove byteLength and byteConstantLength constructor arguments from WasmBlockNode .
Add support for Android Intent URIs encoded in a barcode
ArrayCopySnippets : small cleanup
Correctly skip OpenSsl * tests if OpenSsl . isAvailable ( ) is false .
Simplified lowering phase . Removed " deferred " lowering . Removed custom setLastFixedNode method .
Fix python zip file name when target name has dot ( . )
fixed NPE introduced by GR - 5435
Rethrow the RejectedExecutionException when a worker task is rejected so the caller of executeBlocking is notified - fixes
Truffle : implement language agnostic OSR support .
[ GR - 22784 ] Intercept SubstrateTruffleRuntime . log via TruffleFeature . Support .
added more logging as more people seem to be suffering from host name detection issue .
1 . 1 . 34
for , refactor OpenTracingSQLExecutionEventHandlerTest
Connection view menu enhancements
core : Populate metadata . writtenBy for pre 1 . 3 index files .
support date format ' millis ' & ' unixtime '
improved kotlin emptySet support
Enrich TypeMismatchException for controller method args
convert weight functions tests to unit tests
Import replicated closed dangling indices ( )
Add precise exception for BuildEventArtifactUploaderFactory # create .
Fields in ReplayingDecoder does not need atomic nature . See
execution service : Ignore watch records that where deleted while awaiting for execution .
Make a field volatile and improve javadoc .
partial SPARC assembler
test : verify whether templates exist in an assertBusy ( . . . )
Do not care about predecessor order when duplicating nodes .
MINOR : Fix message count for sliding windows test ( )
Adapt to core Strings method rename
SRC - 202 again . " But Bullwinkle , that trick never works ! " " This time , for sure ! "
Ensure share clipboard sends String not SpannableText
Explicitly enable TrackNodeSourcePosition for test
Fix bugs in getPath ( ) with arrays of objects and arrays of arrays
Move coverageSupport and coverageReportGenerator from BuildConfiguration to
added support for a prefix wild card ( * . field ) in includes
The red test case ( about executorsAmount that is not enforced ) is now green
Fix formatting .
Polishing contribution
Make Array copy intrinsifier active again , fix format string in canonicalizer
[ GR - 10006 ] More thorough testing of proxy connections , do not use proxies that return 5xx codes .
Tests : Remove AwaitsFix annotation and increase logging
BAEL - 16792 Slice 3 | The top 100 articles should have their own package in the module ( )
Fix a match bug .
- Correcting comment in TtmlNode # getCues ( )
Fix handling of IOException when computing key for tokens ( elastic / x - pack - elasticsearch )
[ ML ] properly nesting objects in document source ( )
[ JENKINS - 35184 ] Servlet API dependent bits in a inner - class ( )
BitmapPool - > ConcurrentBitmapPool
Project explorer view
send password 3 times so that the wrong password will cause sudo to exit
Simplified HttpContentDecoder using the new methods in CodecEmbedder
Remove accidental ' + ' in toString ( ) .
Allow to use WebSocketClientHandshaker and WebSocketServerHandshaker with HttpResponse / HttpRequest
Issue - Disc cache directory is changed after app reinstall
Jump by one page when scrollbar clicked .
[ GR - 10956 ] Added test for InvocationPlugins . isEmpty .
[ TEST ] Order of the results is not important in this test
Delegate to common ClassUtils . getQualifiedName
Migrated com . oracle . truffle . regex to AbstractTruffleException .
Makes getDependencies_noTimestamps not date sensitive
Add option to add entries in an ActionFile to DownloadIndex as completed
Updated the Market URLs we use to conform to :
KAFKA - 8928 : Logged producer config does not always match actual configured values ( )
Optimize PooledByteBufAllocator
PRev delimiter detect fix
Added new samples
Add color , dimen , and drawable to reflect
Adapting RestRankEvalAction to changes in API on master
bug fixed for parseFloat / parseDouble , for issue
[ FLINK - 5153 ] Add test for YARN application tags
fix issue that test compile failed
Plugin must be 0 . 20 . x compatible ( tests fails )
Database to database transfer fix ( document attributes resolve )
update XPackRestTestCase to use low level RestClient
Cleanup cast methods for SkylarkList
Testclusters logging improovements ( )
Fix playback for Vivo codecs that output non - 16 - bit audio
Annotate generated constructors and methods with HostAccess . Export .
Minor fix in example .
HUDSON - 3532 : [ FIXED HUDSON - 3532 ]
: here is a StackOverflow bug ? dubbo version 2 . 5 . 9 , FailsafeLogger . appendContextMessage ( ) > NetUtils . getLocalAddress ( ) ( )
SRC - 323 , TestCookie should use real URLs
0 . 19 . 9 no longer supports / _all / _status , closes .
[ GR - 9456 ] All dumpers use the latest version of BIGV protocol
Pass viewport dimensions in drawee
polishing
Fixed KMeans example
[ FLINK - 20045 ] [ tests ] Let TestingLeaderEelctionEventHandler wait until being initialized
refactor , and added test case for verifying the thread pool properties . problem : metricsRollingStatisticalWindowInMilliseconds , and metricsRollingStatisticalWindowBuckets wouldn ' t be set correctly
Fix the 0 sample count ctts entry bug
Ellipse trim path works
delete getColumnName ( )
Don ' t refer to the bazel_tools repository for the iOS provisioning profile
[ FLINK - 15355 ] [ plugins ] Added parent first patterns for plugins .
Showcase app : Replace fragments instead of adding them
fix RuleConfigTest error
Support type exclusion strategies .
Added a simple histogram that can be used in snippets .
[ FLINK - 7314 ] [ futures ] Replace Flink ' s futures with CompletableFuture in TaskManagerLogHandler
Truffle - DSL : optimized ElementUtils . typeEquals
[ GR - 2444 ] Rewrite Truffle tests to not require background compilation .
fix issue , with and without asm
Fix linkifying in selection mode
Tolerate null besStreamOptions when crashing .
Expose getPlugins ( ) to Starlark .
[ GR - 19636 ] SVM ' s System . arraycopy throws IndexOutOfBoundsException , instead of ArrayIndexOutOfBoundsException .
Modify nested classes in ShardStateAction to be static
Run Utils tests on host where possible
Invoke Flyway # setCallbacks only if necessary
DanielWagnerHall for AnshumanSrivastava : Fix deadlock condition in AndroidDriver . I wonder how many more of these we have .
java : Implementing ability to switch to a frame by WebElement in marionette
[ ] Fix incorrect logic to detect if ChannelHandlerContext is writable
Local Gateway : old global state files are not properly deleted , closes .
We cannot get @ Subscribe annotations from anonymous classes , so fail fast
Test the Gradle Plugin against Gradle 5 . 5
[ GR - 10553 ] Use Truffle logging for Context pre - initialization .
Show snackbar ( to provide feedback ) when non - checkable chips are clicked .
Add a @ EnableWebSecurity if it looks like the user needs one
release 0 . 90 . 0 . RC2
[ FIX JENKINS - 40053 ] : Use english as default - locale ResourceBundleUtilTest # test_unknown_locale ( )
2 . x : Expose the Keep - Alive value of the IO Scheduler as System property . ( ) ( )
[ hotfix ] Fix the SingleThreadFetcherManager to get the running fetchers correctly .
for , move RuleDefinitionFileConstantTest from sharding - core - parse - test to sharding - core - parse - common
add testcase & bug fixed for sql - wall
SimonStewart : Reformatting .
RSV column filters UI fix .
[ GR - 19310 ] Close output streams in profiling instruments .
Update NinjaBlackBoxTest . testDuplicateNinjaPartsDifferentMappings so that it is resilient to different error messages .
PG : object search fix ( schema filters )
Cipher suite conversion between Java and OpenSSL
removed the static field in ChildrenConstantScoreQueryTests
remote : fix race on download error . Fixes
Add fall - back behaviour for vertical cues if rendering isn ' t supported
Make one of the ExoPlayerTest tests more sensible .
Fix toolchain deps for - - noimplicit_deps cquery .
Refactor XABackendDataSourceFactory = > DataSourceMapConverterFactory .
Resolved the Comments
Polish alibaba / nacos : more generic implementation
Use lazy class loading instead of explicit assertionsEnabled check
Improve How to Count the Number of Matches for a Regex ? ( )
[ ML ] Report correct count for df - analytics get - stats API ( )
MINOR : Remove Diamond and code code Alignment ( )
Http2ConnectionBase should flush directly the connection using the ChannelHandlerContext and not use the parent class flush
Add MockMvc - based integration tests for management . security . enabled
Prevent NPE in ImaAdsLoader onPositionDiscontinuity .
Dynamic VBO ' s = slow . Change from VBO to VertexArrays gained 12ms with 50 animated models .
Polish
chore : fix some issues reported by sonar
Fixes . @ One is broken in 3 . 2 . 6
Geo search across anti - meridian doesn ' t work , closes .
[ FLINK - 17017 ] [ runtime ] Allow nullable timeout for streaming slot request in slot pool
SQL formatter order ( alphabetical )
Make parse_header a validation action
[ FLINK - 14433 ] [ DataStream ] Move generated Jaas conf file from / tmp directory to Job specific directory
ValueViewPanel fixes & actions
SimonStewart : the augmenter should copy fields from the existing driver to the newly created one .
Remove ` jenkins . setNodes ( jenkins . getNodes ( ) ) ` style hacks to force the master number of executors to be updated
modify the cat queue size
am cbb0b14c : am dace5eb5 : am 3814cf37 : am 80b7fc1f : Merge " Use Button for Snackbar \ ' s action " into lmp - mr1 - ub - dev
modify data source parser and factory .
Minor typo fix : no logical changes
Improve DoS test case to be more robust to timing
adapt _cat / nodes to node . client removal
[ FLINK - 2331 ] Whitelist the ' akka . remote . RemoteTransportExceptionNoStackTrace ' to be acceptable in YARN logs
Issue - " Can ' t create " . nomedia " file " logs
Send STOMP ERROR if external broker not available
Adapt to changes in master
Fix path in GlobalMetrics . openPrintStream
Actions : correctly fill sequences
Fix a NPE while trying to compare two data bound beans with null
bug fixed for List < String >
Allow Array . set ( objArray , index , null ) to not throw NPE .
Revert code to avoid NPE in RPC wire after providers are cleared .
[ GR - 10260 ] Reduce the number of boundary calls in AllocationReporter .
decrement jar size
Fix warning .
[ hotfix ] Minor code cleanups in AbstractStateBackend
[ JENKINS - 36476 ] Underprivileged users could not use the default value of a password parameter ( )
use Collection
Fixed - zRemRangeByScore and zcount methods of ReactiveZSetCommands don ' t use - inf and + inf values .
Add TextureFilter to parameters for TmxMapLoader .
Removing lambdas .
Searchable snapshot number of prewarmers ( )
Allow if filtering in list comprehensions
Fix handling of cyrillic characters in AsciiBytes hashCode method
relaxed test conditions
RELNOTES : Allow java_lite_proto_library in the deps of android rules .
Environment variables usage info labels are restored .
Tasks wizard refactoring
More and better value setters in GraphBuilderConfiguration
wait before closing the shard if a snapshot is still in progress
Add @ NullableDecl on the field in TreeBasedTable .
args4j sets null as value when the value is " "
fix a bug
Put safeguards around Recipient creation in the IdentityStore .
Polish CF management skip SSL opt - in
Create libpath according to operating system .
Debug : update for recent changes in stack iteration
HLS : Fix slow seeking into long MP3 segments
ERD paste handler fix
- renamed RuntimeExceptionFailureITCase to TaskFailureITCase
[ GR - 24618 ] Fix access privileges were not recomputed for the first preinitialized context .
avoid repeated maxDescrLength - ending . length ( ) calculations in getTruncatedDescription
Add NewInstanceServiceLoader . newServiceInstances ( ) .
Refactor freemarker directive .
Proposal for same site cookie value
fix for IfNode reordering
[ GR - 10409 ] Workaround for a Chrome URL bug .
[ GR - 4262 ] Fix compilation error in LLVMStack . allocate .
add getLogicAndCipherColumns ( )
Handle case in new earliest schedule when end node and begin node are identical .
trivial : removed unneeded public modifier in interface methods
[ JENKINS - 47056 ] nested JNLPLauncher settings ignored
Add random positive time value convenience method
Fix EngineJobTest
Loader changes
Followup API change for elastic / elasticsearchelastic / elasticsearch
RedissonSetCache TTL not updated if element already present
bug fixed for ' is ' prefix fieldName smartMatch . for issue
[ FLINK - 1464 ] [ java - api ] Added ResultTypeQueryable interface implementation to TypeSerializerInputFormat .
fixed issue 1258 , Intersector . intersectRayBounds was not public
Test against EhCache 3 . 0 M2 ( as a JCache provider )
[ GR - 3227 ] Simplify and improve address lowering to swap index and base when appropriate .
[ TEST ] Fix broken test that still tried to acquire the shards to set it non - idle
Add comments on the two new match .
remove accidental radius parameter from fillTriangle
RSV status line color fix
Nested Jars contained in subdirectories are now supported .
DouniaBerrada : Fixing BrowserConnectionTest on Android .
Update AbstractItem . java
fix demo ad
Tests : Using UTC time zone in RangeQueryBuilderTest to make forbidden apis checker happy
Give the profile writer thread a name
fix user search
removed simple OR
Reformat EdgeOptions . No logical changes .
findKeysByPattern response interface changed
Add unit test for MySQLBinlogEventPacketDecoder
[ FLINK - 12641 ] [ coordination ] Release partitions on every terminal job state
GT - 2972 : Fixing MapLoader ( closes ) .
H2C upgrade failure HTTP / 1 . x response should not be processed by the server handler - fixes
- fixed TPCHQuery3
Consolidate prerequisite handling in CcCompilationContext
Don ' t attempt to load camelCased version of the package name .
A little bit of optimization
for checkstyle
Change Timeout . cancel ( ) to return a boolean value , true on a successful cancel
Add / error to ignored paths for security autoconfig
Throw translog corrupted exception on malformed op
Use color int instead of ColorStateList for box stroke color .
Actually change FAB elevation on API 21 .
fix : rename test
SSH tunnel re - initialization fix ( keep old port number )
New formatter settings for enum values ( every value on a new line ) .
, release JDBC resources when the channel become inactive
Warnings .
typo
Oracle session viewer - add filters
consistent naming
core : rename wrapped synthetic method ( )
Fixed builder pattern port
Connection page switch fix ( slow dialog render )
[ hotfix ] [ runtime ] Make slotSharingGroupId nullable in TestingLogicalSlot
[ java_common . compile ] Name output source jar relative to the output jar name
Fix
- Implemented getTextInput in IOSInput with UIAlertView .
Fixed markdown render with short url .
[ FLINK - 15240 ] [ hive ] is_generic key is missing for Flink table stored in HiveCatalog
, refactor QueryResponsePacketsTest
Preparing to include transitional release .
DanielWagnerHall : Reverting added logging from revision 14551 until logging control is added , see revision 14589
[ hotfix ] [ kinesis ] Fix inproper test name in FlinkKinesisProducerTest
[ TextInputLayout ] Fixed wrong hint position when an error is shown and field is empty and unfocused
Composite objects reading fix
add action for downloads menu item
Log port ( s ) used by Undertow
Fix checkpoints in assertion
arthas - boot support versions option .
Fix http : / / b / 73496081 which was caused by not filtering on aliasconfiguredtargets that are actually ruleconfiguredtargets .
GT - 2187 - tweak error messages
[ FLINK - 17506 ] [ state - processor - api ] SavepointEnvironment should honour ' io . tmp . dirs ' property
DefaultPromise internal state dependent on Signal
[ hotfix ] [ tests ] fix codestyle issues in ResultPartitionBuilder
Update isSymlink to be closer to the current version in apache .
Simplified IdentityRootAction . getFingerprint
Avoid starting RequirementsWatcher if there is no download task
[ FLINK - 2825 ] FlinkClient . killTopology fails due to missing leader session ID
flatMap overloads
Fix reschedule async fsync test
Fixed , enhance netty support for rest protocol
Improve WebClient test with ParameterizedTypeReference
Refactor integrate hint test cases for prepared statment .
Fix corner case for javadoc redirector
Fix compilation error
[ test ] if enabled always specify setting other randomly specify setting to false .
Ensure IovArray is usuable without sun . misc . Unsafe ( )
Fix small colorFilter issue
Recognize more counted loops
2 more try - with - resources ( buffered + close )
Bael - 3893 AWS AppSync with Spring Boot
Passing execution id over the libgraal boundary .
Make Maps . toImmutableEnumMap use j . u . f . Function .
Polish
Mime type string always finish with a single 0 byte
Profile the condition in IfNode .
Polish
Minor refactoring .
Preference pages layout fix
allow to dynamically set cluster . routing . allocation . cluster_concurrent_rebalance using cluster update settings API
MySQL : table filter include + exclude criteria fix
[ FLINK - 16060 ] [ checkpointing ] Generalize LinkedBufferStorage to support arbitrary number of linked buffer storages
Protect BytesStreamOutput against overflows of the current number of written bytes . ( )
Refector ShardingJdbcDatabaseTester , connection cache add schema to support Oracle
Fixed crash with NullPointerException when set value before laid out in Slider .
[ fixed ] added isAnimationFinished back to Animation class , i suck . Back to my broken bath room / kitchen . . .
fixed ddl columns default values
Reindex : Fixed typo in assertion failure message ( )
Disable SQL expressions result resolve ( to table columns )
Minor changes after review
fix permissions for REST tests
Ensure that jar entry names use forward slashes , even on Windows
No logical change : remove unnecessary generic parameters
avoid publishing unadopted children in node replacement to fix potential race
Allow - - add - opens like command sequences for custom JavaArgs
more context for exceptions within DebugInfoBuilder
Improve VarArg handling in sulong
disable console printing from unit tests
Forbid setProtocols with HTTP_1_0 .
default value information of serial types , unnecessary conversions to serial types removed
Validate that test & lifecycle methods are not @ Autowired
increasing the time out as I ' m seeing errors like this , which is legitimate :
Added more thorough test for primitive DataTypes ( Fixed )
Minor changes .
prevent string concatenation if log level is coarser than FINE
Call onKeysRemoved ( not onKeysRestored ) when releasing offline keys
[ FLINK - 18403 ] [ checkpointing ] Ensure that unaligned checkpointing is only activated for EXACTLY_ONCE .
framestate invariants : style fixes
text / plain will make the debugging easier as you won ' t have to go to the source view .
fix the issue that an immediate login after logout will be redirected to signin page ( )
Avoid assertion failures for BoxedVirtualObjectNode
Remove legacy index . query . bool . max_clause_count did - you - mean ( )
SimonStewart : Removing some printlns
Polishing
Fix missing suppress warnings .
parseEncryptorRulesConfiguration ( )
Fixed longpress - drag issue .
Add update site management pref pages
Add a new state of ` terminated ` to fix the race condition of ` errorEmitted ` ; Add more comments .
[ java ] Replacing SimpleDateFormat with a more modern class DateTimeFormatter
Add asFixedNode to FloatingAccessNode so that GuardLoweringPhase supports any type of FloatingAccessNode
Remove references of ConfiguredTarget . getTarget ( ) from ToolchainUtil .
Open SQL console handler fix ( NPE )
Read - only text background fix ( not really )
Simplify prev assignment in ContextThreadLocal .
added onJobChange ( ) so that views can reflect item changes
SourceValueFetcher should check all possible source fields ( )
[ hotfix ] [ table - api ] Add literal conversions between all numeric types .
[ FLINK - 19816 ] Make job state cleanup dependent on final job result
Clean up naming of demo app download components
[ GR - 25735 ] Handle deoptimization in all CompiledMethodTests .
Don ' t print warning when skipping RIFF and FMT chunks
Fix bug recently introduced to CppCompileAction . discoverInputs ( ) . While in a
Add content encoding detection for downloader
PatrickLightbody Don ' t assume String is returned , since the internal execute can return Boolean , Number , Void , etc .
It appears sorting based on y up / down is not necessary .
DUBBO - 48 accesslogFilter NPE
Truffle : added support for Runtime # getCallTargets ( )
Handle void type when resolving a signature type .
0 . 2 . 2 - SNAPSHOT
Change page to sheet .
Mapper : Upgrading a simple mapping to multi_field mapping fails on merge conflicts , closes .
Polishing
Fix problem with direct share
[ ISSUE - ] Optimize notifyCenter code ( )
Remove old check . Replaced by HostVM . checkType ( ) .
Added null check suggested by Kohsuke
Modfy datasource maxActive to 1 and fix other temp table code bug .
Added first unit tests for Nephele client package
, test cases
Only update shape paths once per draw
remove RealDealIntegrationTest , because it starts its own server , and is thus incompatible with ClientDriverSuite ( which also starts a selenium server )
SimonStewart : Introducing a rampantly ugly hack to try and make the Firefox driver more stable on OS X
Update Settings : Changing the number of replicas does cause allocation / deallocation of shards , closes .
MemSchedTests : disable implicit null checks
Removed the workaround for multi - lock namespace definition
[ fix ] ControlFlowGraph : Fix deferred exit traversal deferred loop computation .
Do not use Graal as a JIT compiler for native image generation
KAFKA - 6166 : Streams configuration requires consumer . and producer . in order to be read ( )
Adjust hpack impl to be compliant with draft 3 examples , and enable http2 in MWS .
do not rely on node source positions for showing language call stack at PE bailouts
for : refactor DatabaseUtil
Remove unused make variable GLIBC .
[ added ] Support for utils collections to use a typed backing array .
HTTP / 2 Allow more time for EventLoopGroup to shutdown in test that use LocalChannel
Remove the ex which not thrown . & Remove unused logger . ( )
[ FLINK - 10785 ] [ tests ] Upgrade FlinkKinesisConsumerMigrationTest to test restoring from 1 . 7 . x
Remove 2 deprecated SimpleExoPlayer constructors
To keep JDK 5 compatibility , 1 ) removed throws clause and 2 ) create an internal constant for global logger name .
reuse same datasource connection when proxy - mode is connection_strictly .
Made the ` ticker ` engine the default engine .
Ignore script fields when size is 0 ( )
[ FLINK - 14522 ] Introduce JavaGcCleanerWrapper to find Java GC Cleaner depending on JVM version
Drop separate user dest property for subscriptions
[ hotfix ] [ rest ] Introduce MinMaxAvgStatistics . valueOf helper method
Let to push lambda arguments not only with * LOAD instructions but with SIPUSH and * CONST_ * .
Fix possible NPE in FakeClusterManager
Polish " Stop limiting layer customization to external modules "
WallFilter support OracleAlterIndexStatement . issue
[ Snackbar ] Add support for passing a Context to Snackbar . make ( )
Test ImmutableMap . equals in ImmutableMapTest . testEquals .
Resize image in attempts to get it to fit into the maxImageSize bytes .
[ ML ] Remove JobManagers dependency on JobResultsPerister ( elastic / x - pack - elasticsearch )
Fix test case error
Delete and expire MultimapCache methods fixed .
added a test case for JNLP launch
complete set grammer in PostgreSQL
Minor modification BAEL - 835
Undo misuse of isAttachedToRootTag .
modify patter for mysql
Restored a corrected Reed - Solomon optimization and fixed a small issue in black box test cases
Align WebFluxAutoConfigurationTests with breaking Framework change
Use standard location for errno . h inclusion
[ java ] Fixing deserialization from JSON for RequestIntercepted event
handled combine functions that throw exceptions and added a test against this case
make FixedNodeProbabilityCache behave better in the presence of dead code
[ hotfix ] Move FileBufferReaderITCase from ' flink - tests ' to ' flink - runtime '
Disallow to return some settings that aren ' t in theme configuration
MySQL / MAria : permission management fix
Remove throws AssertionError .
[ GR - 17680 ] Retried failed compilation should TrackNodeSourcePosition .
[ streaming ] Added NoSuchRecordException and StreamRecord tests
Geo : add Geometry - based query builders to QueryBuilders ( )
[ FIXED JENKINS - 29568 ] A better fix
DouniaBerrada : Updating TouchScrollTests to work on Android tablets .
Use Truth assertions instead of JUnit assertions in InetAddressesTest .
Replace JavaConstant . asObject calls with ConstantReflectionProvider . read * variants .
if more rule decorate metaData bug fix . ( )
Fix formatting .
[ GR - 20688 ] Make FloatingReadPhase code size increase check more tolerant .
modify the test
Correctly show the default mode ( - - jvm / - - native ) in the - - help output with the Bash launchers
fix : don ' t stop loading classes in case of an error ( PR )
[ hotfix ] [ task ] Extract InputSelection . NONE_AVAILABLE constant
- - mistake
typo
[ GR - 19217 ] Cache the guest object in DebugValue .
Adapt to changes on master
Remove @ Component from ResourceServerFilterChainOrderProcessor
Crashing instead of ignoring unexpected exceptions .
fix assert in HotSpotResolvedJavaMethod . vtableEntryOffset
improved MapDeserializer compatible
Improves AbstractWireSerializingTestCase equals test ( )
changed first input on bin ops to be reg only
oracle 10 and 11 constraints fix
Use the indirect call node in WasmFunction # execute .
Renamed square setting , made packing try to use smallest square size .
Decrease the level of logging in WebSocketFrameEncoder / Decoder ( )
Extract factory method for FragmentedMp4Extractor in DefaultHlsExtractorFactory
Suppress less checkstyle checks in tests ( no need for javadocs )
BenchmarkCounters : make thread - safe and refactoring .
Issue : HUDSON - 2494
Allow overriding of setOutputSurface workarounds
Remove only - in - tests null checks in SkyframeActionExecutor .
- Fixed copy - and - paste mistake in exception statement
[ tests ] Increase timeouts for process failure tests in order to compensate for volatile CI environments ( Travis )
ByteBuf . readBytes ( . . . ) should use the allocator of the buffer to create the new buffer . Related to [ ]
Make sure JsonArray is mutable when creating with array
Give better names for String dumping methods .
Fix typos and style
Fix DashDownloaderTest . testDownloadManifestFailure
Fix can ' t find job error ( )
Update FillFormatter with new interface
Fix a race when creating two ZIP files with the same name
Warn instead of crash if setColorFilter is used .
BlockingQueue . peek race - condition fixed .
[ FLINK - 14766 ] [ coordination ] Remove volatile keyword in executiongraph package
Add ServiceLoaderUtil to improve SPI classloader mechanism and update SpiLoader
2 . x : Improve Observable . takeUntil ( )
KAFKA - 2571 ; KafkaLog4jAppender dies while specifying acks config
Reduce Lettuce shutdown timeout to 100ms
if a module is not linked from POM any more , then don ' t let it participate to the dependency tree .
Improve prekey refresh logic .
[ hotfix ] [ network ] [ tests ] Refactor existing tests for reusing PartitionTestUtils
fix fugly failure message on put index template
Relax the sanity check in HttpClientUpgradeHandler
Fix handling of generic options with arguments .
Make Truffle compilations multi - threaded on multi - core systems . Introduce TruffleCompilerThreads option for a manual override .
Don ' t use Skyframe globbing when we ' re not tracking incremental state .
[ FLINK - 10774 ] [ tests ] Refactor FlinkKafkaConsumerBaseTest # testConsumerLifeCycle
Create our own dialog when GooglePlayServicesUtil can ' t .
fixed format
minor fix
fix : remove redundant cast over similar conditions ( PR )
Throw exception if performance warnings where seen ( with tests ) .
[ grid ] Invoking the create method from MemoizedConfig instead of Config
Improve AbstractQueryTestCase # unknownObjectExceptionTest ( )
Automated rollback of commit 047688a129b444ba918ae4c31767dce73b17b049 .
[ FLINK - 14784 ] [ table ] CsvTableSink miss delimiter when row start with null member .
Oracle NULL / NOT NULL alter fixed .
Ensure locking for event based rewrites ; reload state and exclude when doing that .
MINOR : Fix ConcurrentModificationException in TransactionManager ( )
[ TEST ] Update strings in ML categorization test ( elastic / x - pack - elasticsearch )
More tests
- fixed job graph generator : Reducer gets key columns and classes for local strategy NONE
http : / / team . baeldung . com / browse / BAEL - 11412
sulong : make symbol renaming marker unambiguous [ GR - 21598 ]
Fixed boolean coercion ; made boolean constants immutable
performance improvement in the recursive copy method .
Update CoverageReportActionBuilder to use SpawnStrategyResolver .
Fixing HSL to RGB converter in Java
Objects # requireNonNull guard in MessageDigests
moving unit tests from . testsuite - > . beans
correct ServerConfigManager
Incorporate latest mozilla tld changes .
SQL : Enable adding missing equals to bool fields as filters ( )
[ ML ] Fix race condition between multiple job deletion and immediate recreation ( elastic / x - pack - elasticsearch )
[ FLINK - 5160 ] SecurityUtils use OperatingSystem . getCurrentOperatingSystem ( )
use scaled addressing in preference to unscaled where possible
DanielWagnerHall : Fixing thinko
Make Aar provider into NativeInfo
Refactor ` Target_sun_util_locale_provider_LocaleProviderAdapter ` into implementations for JDK - 8 ( or earlier ) and JDK - 9 ( or later ) .
Polish
Add HlsDownloadTest
Drop AuthorityReactiveAuthorizationManager and avoid need to block
Entity ERD fix ( load referencing entities )
Cluster ' s cascade slaves support
Rename field to clarify function .
Security : propagate auth result to listeners ( )
Treat intersect masks like add masks ( )
Fix ordering of keys in PropertySourcesPropertyValues
Add a null check .
Always include initial handshake exception when throwing SslHandshakeException ( )
Resolve conflicts
Alerting : Small fixes .
Minor issue fix
Add unit test for YamlOrchestrationEncryptDataSourceFactory
Fix typo in exception message : quite - > quiet
Make messages async by default
Fix Cea608 caption valid
improve prefer local execution
[ hotfix ] Fix checkstyle violations in ExecutionGraphDeploymentTest
Adapt device change to new springcli . properties
ThreadPools : schedule a timeout check after adding command to queue
Fix the R . txt to generate a transitive closure instead of the current resources .
Polish contribution
[ GR - 2362 ] Skip unnecessary Pis when lowering array length
Fix ReferenceCountedOpenSslEngine . getEnabledProtocols ( ) when using boringssl
Improved log message in EC2 cloud manager
[ Test ] testGetFields_complexField should wait for a green cluster
make netty threads daemon
Extend support of 608 captions in SEI messages
clear ( )
[ hotfix ] Make SlotPoolBuilder a top level class
[ FLINK - 1201 ] [ gelly ] reverse and graph input methods
Update App . java
Combine initNotificationChannel ( ) and setUpUpdateNotificationChannel ( ) into a single method .
TexturePacker $ Settings now correctly copies scale and scaleSuffix arrays from another instance ( )
Support PATCH requests in MockWebServer
Fix API usage after index - level guice was removed
Implemented LotteAnimatableShapeValue but not tested
Upgrade API : add common upgrade check method ( elastic / x - pack - elasticsearch )
Columns viewer - repacking fix
Improve test suite .
Bypass local cache at client start by default .
add datasource parameter converter .
Allow AudioDecoderTrackRenderers to override their output formats .
Add null check
Remove value equality check for proguarding
Fixed - NullValue object shouldn ' t be stored if RedissonSpringCacheManager . allowNullValues = false
[ inlining - 4 ] getAssumptionInlineInfo ( ) can get context . getReplacements ( ) itself
Enforce sane upper limit for TTL in DefaultDnsCache . ( )
Automatic Extract : before ( ) in org . junit . runners . model . Statement # evaluate
Use AVERAGE - BANDWIDTH instead of BANDWIDTH when available
Fix crash when test_suite includes test in broken package , with - k
Ensure - - experimental - build - server does not affect - - dry - run
TraceRA : add TraceBuilderResult support to CFGPrinter .
Parse CHANNELS attribute from EXT - X - MEDIA
RedissonLock added
sulong : add lli - - toolchain - api option
Add skeletontest
[ GR - 15742 ] Fixes for native - image - agent build and restrict mode .
Reorder options so TraceCompilation and TraceCompilationDetails are next to each other
Allow StaticAnalysisResultsBuilder customization
[ FLINK - 17258 ] [ network ] [ test ] Run ClassLoaderITCase # testDisposeSavepointWithCustomKvState without unaligned checkpoints
[ TEST ] Give time for results to be searchable in UpdateInterimResultsIT
Add nullPointerException comments and ObjectHelper test code . ( )
bug fixed for oracle sql parser .
[ GR - 26049 ] Fix PolyglotContextImpl # getLocal to use correct stable locations reference on the fast path .
DanielWagnerHall for AjayKemparaj : Fixing HtmlUnit select element getAttribute behaviour for multiple . Fixes issue 2158 .
support Generic in public - field and setter - method
Fix HTTP / 1 client bytes written metric regression
Fix for ExasolConnection Manager
Skip the execution of an empty pipeline ( )
Table resolve fix
Fix testHideShow on pre - ICS
config to json 3rd version
refactor JDBCXABackendDataSourceFactory
modify test cases for rowCountToken
WebDriver - backed Selenium tries to emulate Selenium Core ' s alert handling by
PG dump : case - sensitive schemas backup fix
add more wall error info
[ Test ] Prefer ArrayList over Vector ( )
Improve test output .
Fix build after JDK - 8235521 Unsafe . * Initialized changes
[ hotfix ] [ core ] Make sure Dropwizard reporters do not block operator creation / teardown while reporting
Multiple key events handle fix
Stopped sending the decode duration in a message , which no one was looking at , and logged it instead .
Remove by index methods for ordered map and set .
[ GR - 23871 ] TRegex : fix predefined whitespace character set in Python ASCII / Bytes mode .
unwanted change went for ResultSetLogger , back to Jeffs version
TomcatInstrumentableClassLoader supports Tomcat 7 . 0 . 63 + as well
[ ML ] [ Inference ] Unify top_classes object field names with analytics ( )
[ GR - 6636 ] Allow raw pointers to be passed to NFI closure typed arguments .
removed js related piece of logic from the ICU4JFeature
Also add inputs when adding node returned from InstanceOfNode . create method .
modify frontEndException alert frequency
TNS URL redesign ( use TNS descriptor )
[ GR - 18831 ] Use static JNI libs also for Java 11 .
BAEL3889 - Adding braces in EvenOddPartitioner
File report if rewinding fails and an action observes a lost input twice
Blank line delimiter handle
[ GR - 20876 ] Fix merging issues for escaped large byte array accesses .
Revert " Do not suppress channelReadComplete ( ) when a handler was just added "
Fix playback speed in MediaSessionConnector
Access content of the loaded source
Modify test cases for " digest " in LeafSegmentKeyGeneratorIT .
Bulk API : _version on delete actions is not honored , closes .
Show upload progress for view - once messages .
[ java ] Minor performance improvement
Fix compile due to change in InternalSearchResponse signature
null AssetFileDescriptors support added in ` ContentDataSource `
Connection edit wizard pages reorg
Use serializable exception in GCP listeners ( )
Truffle : cleanup to the LoopNode API .
A couple of fixes enabling Sulong / FastR
Refine AbstractJarFile method visibility
rename to isUsingRegistry
[ hotfix ] [ network ] minor improvements in UnionInputGate
Rework test case so that the pass does not have / / and work on OSX
Only use Map converter binding when no descendants
GlobalEventExecutor usage removed .
fixed client - adpater logger test
Tray icon behavior enhancement
Reintroduced set for recently removed channel ID in order to make cancelling of tasks more robust
polishing
removed unecessary dependency on ServletException
HOTFIX : fix broken streams test
Added a few guards and asserts .
MINOR : don ' t log config during unit tests ( )
added DebugVerifyHandler implementation to verify all phases delete dead nodes
Fix for http : / / code . google . com / p / mybatis / issues / detail ? id = 307
Removed 2 classes for main and other for Runner . Replaced them with a single class serving both the purposes
Removing a trailing " ? " - symbol in the end of the URL ( IE6 and Chrome add
refactoring
[ java ] Small refactoring of debug server
[ hotfix ] [ runtime ] Make SchedulerBase # getSchedulingTopology ( ) final
LinearScan : make LSRAOptimizeSpillPosition a NestedBooleanOptionValue .
Rolling back change because of subtle race condition when multiple actions insert ActionInputs into SingleBuildFileCache .
Clean up BazelTestSuiteRule .
Prototype Truffle inlining when FastPE is enabled .
[ TEST ] generate random timezone out of the available ones in joda
fix formatting
update checkstyle for Class Design
Function / Lambda parameters do not need to be lenient . . .
incorporate feedback
Remove EventHandler from SkylarkCustomCommandLine .
Save task action ( label fix )
Fix dispatch of play when in IDLE or ENDED
Fixed - shared session between several Tomcats expires earlier if readMode = Redis and broadcastSessionEvents = false .
Animate FAB elevation for new hovered and focused states .
Don ' t apply a host - configuration transition to the module_map attribute .
XML parser fix ( element body extraction )
HLS : Better stream selection at start of playback .
Upgrade to Neo4J OGM 2 . 1 . 3
Change Locale . getDefault ( ) to Locale . US , us for url is better .
Update class name in log messages after recent rename .
LwjglAWTCanvas no longer ignores system repaints with noncontinuous rendering .
Revert " KAFKA - 9533 : ValueTransform forwards ` null ` values ( ) " ( )
Extracted ' constraints ' methods and renamed variables ( )
Fix string coercions in tool tests .
revise : use ShardingConditionAlwaysFalseException to decouple OptimizeEngine
Javadoc typo
TAB - > WS
Don ' t register processors in # after ( final EventProcessor . . . processors )
whitespace , javadoc typo fix
[ TEST ] Opt out of dummy documents where scores are relevant .
KAFKA - 7477 : Improve Streams close timeout semantics ( )
Added missing colors .
SQL formatter : unclosed quotes parse fix
Make use of a Logger in the SPDY example and simplify it a bit
[ ML ] Update DFA progress document in the index the document belongs to ( )
OperatorAsObservable
Code improvements .
Allow register debug info in all reference maps .
[ FIXED JENKINS - 23970 ] CLI authentication with username / password try to read my . ssh / id_rsa and prompts for key password
Checkbox tree fix
MINOR : Follow up for KAFKA - 6761 graph should add stores for consistency ( )
Add test case based on SPR - 16615
Support RightToLeft direction for legend
In SequencedSkyframeExecutorTest # testThreeSharedActionsRacing , make sure action C has started before interrupting the main thread .
[ FLINK - 13631 ] Update FlinkKinesisConsumerMigrationTest to restore from 1 . 9 savepoint
Remove unnecessary logger creation for doc values field data
Ensure CompositeByteBuf . addComponent * handles buffer in consistent way and not causes leaks
CSV import NULL string mark support
Avoid unnecessary frame allocation in GifDecoder .
Grid column header render fix
[ ] Allow to specify if direct buffers are prefered in ByteToMessageCodec
Truffle : remove FrameTypeConversion interface
[ GR - 21956 ] Truffle : introduce the ShowInternalStackFrames option to include internal sources in stack traces .
Remove the following methods from
GWT fix key mapping for right shift , right ctrl , right alt , numpad star and numpad divide keys ( )
sort code
[ TEST ] wait for green before deleting mapping
Removed message and several unneeded lines
Make local cache creation overridable in RedissonLocalCachedMap
Reduced unnecessary duplicate calls on NioDatagramChannel
Add code missing in initial value handling for options
Make invalid seek positions fail on playback thread only
Remove not needed method indirection .
Provide a better error message for the case when all shards failed ( )
update version
Bump dependency versions to the latest .
SQL query job fix ( avoid NPE during execution )
Catalog / schema selector dialog redesign
Implemented the error handling in case of commucation failures
TraceRA : TraceGlobalMoveResolver : minor clean up .
Make DebugServerTransportTest and SkylarkDebugServerTest work inside macOS ' s network sandbox .
Use more robust check for bitcode inputs to link
refactoring
fix the gzip queue size
fix BalancedBrackets . java
Use a canonical instance for an ActionSketch with null hashes .
Refactor page test cases for oracle .
[ GR - 10451 ] Take correct size of frame in NewFrameSlot and determine kinds correctly .
Fix compilation errors
Fix calculation of age of pending tasks
Tweak FileSystemWatcherTests timing for Windows
[ FIXED JENKINS - 17606 ] Reuse existing fingerprint action if present .
Add list . pop
Transformer presentation selection fix
updated UTC type adapter to support JDK 1 . 6 using Jackson ISO8601 parsing util
speeling
Tests : Additional test for memory stats api .
lambda class name suffixes can use ' . ' as a separator instead of ' / '
Add user test case
[ JENKINS - 45472 ] Copy the list in case it ' s reused elsewhere
fix path seperators in bundle tests
Changed the TermFilter to return a native DocIdSetIterator instead of the FixedBitSet ' s implementation .
Fix joining of Sulong stdout results in RemoteTestSuite
UriComponentsBuilder method to configure URI variables
Description redacted .
Comment out debuging code
Update Devtools to ignore manifest Class - Path entries that do not exist
Fixed default formatters test to reliably work with any system locale
Bad regex in CORS settings should throw a nicer error ( )
fix for http : / / code . google . com / p / mybatis / issues / detail ? id = 499 . rollback cache by default on dirty sessions
[ GR - 26377 ] Refactor Target_javax_crypto_JceSecurity .
Polish
- Unit Of Work : Make final instance variable of immutable Student class .
Make mapping from position to time more accurate in XING MP3 streams .
Remove GSON from the RegistrationServlet
add testcase
Unsuccessful attempt to save r12 when heap base is zero , verification uses it
Improve VMError diagnostic output
Replace remaining stbl assertions with warnings
For , implement getSchemas
Return UnsupportedInlineAssemblerNode instead of throwing error
Fixed a bug with using variables in multiline templates
Propagate output format in tunneling mode
Ensure to release the component when it ' s removed from CompositeByteBuf / Add tests for reference counting of CompositeByteBuf
Fix invalid flag setting for RegExp after lucene upgrade ( )
Unified refresh in database editors
Fix mp3 extraction failure for short frames
Only parse common - encryption sinf boxes
add more test cases
Backed out changeset : e8a56e4376fc
Improve performance of shards limits decider ( )
Make Wait throw its own custom exception instead of a JUnit exception
Incorrectly quoting values of a UUID array
[ scene2d ] Removed x , y parameters for Tree # drawBackground .
[ ML ] fixing minor change in langident model behavior ( )
Use same symbol export approach as on Darwin also on Linux
[ FLINK - 14221 ] [ table ] support drop temp system functions and temp catalog functions
Reconnection handling fixed .
for , refactor Pagination . getActualRowCount ( )
Refactor exception handling in SandboxStrategy .
Fixed incorrect placement of { in Iterables . contains method .
Make JDBC - based batch configurer back off without a DataSource
[ FIXED JENKINS - 16444 ] NPE showEvenIfUnstableOption when first adding trigger section .
Use native functions for getuid / getgid
[ FLINK - 5811 ] [ tests ] Harden YarnClusterDescriptorTest
core : fix sythetic constructor replacement ( )
fix compability issue
Compile fix
modify keyGeneratorProperties
Items order in project fix
Properly expose asset provider for targets without explicit manifest
Ignoring failed tests , regression in marionette . Bug - reports submitted and referenced in the tests .
improved smart match performance .
Looks like the Redis spec has changed
Change ImmutableRangeSet . Builder to reject overlaps
Allow servers to specify ENABLE_PUSH to 0 explicitly
Remove weird nextOffset state from SampleMetadataQueue
MINOR : refactor CandidateState . unrecordedVoters ( )
Stored procedure exec fix
Fix a division method mistake ( )
extracted decision method
ERD persisted state management UI
add properties .
Improve site - example integ test to test served contents .
Fix consistent order of assumption and type getters .
LLVMAsForeignNode delegates to LLVMAsForeignLibrary
Add the possibility to specify the dump files path in ExtractorAsserts
[ FIXED JENKINS - 11110 ] use different attribute names to avoid catch - 22 situation with old Jenkins .
[ GR - 4087 ] Include import for RootNode when the handler of IS_POINTER message is not defined .
Remove unused parameters in AnalysisRegistry ( )
LLVMFunctionSelectNode should specialize on LLVMFunction .
RSV panels refresh fix
[ FLINK - 3835 ] [ optimizer ] Add input id to JSON plan to resolve ambiguous input names .
[ JENKINS - 26411 ] Bring slave log link back
tests : Add to ability for a integration test to prevent specific templates from being wiped between tests .
Making tests fail more meaningfully .
KAFKA - 6802 : Improved logging for missing topics during task assignment ( )
Remove usage of deprecated API
Restore Hibernate ORM 5 . 1 compatibility in HibernateTransactionManager
[ hotfix ] [ table - common ] Fix logging in the table source validation procedure
[ hotfix ] [ yarn ] [ test ] Avoid using Mockito for ContainerStatus in YarnResourceManagerTest .
Speed up getting the compiler configuration when the compiler is not yet initialized on HotSpot .
Polish ExecutorSubscribableChannel
Fix loading MapProperties from tmx file .
Attempt to fix failing Windows test
Fixed - RedissonReadLock by name with colon couldn ' t be unlocked properly
[ GR - 9246 ] Allow arbitrary pointers in tls .
Formatting .
Fix DanglingIndicesIT Failures from MasterNotDiscoveredException ( )
Avoid IdleStateHandler triggering unexpected idle events when flushing large entries to slow clients ( )
Enable seamless rejoing for Vp9 extension .
DecoderState added
Track input / output cleanup .
[ fixed ] use desired width and height instead of constants for viewport calculation
javadoc fixes
Backed out changeset : 00f568f95ffd
Make copyAnimation method public and split it . ( )
add renew ( )
[ grid ] Wrap reading the model in a read lock
Disable chronometer for playback speeds ! = 1 . 0
Polishing
Added a few more tests for DbObjectCache
fixed issue , date format
remove workaround for highlighter bug with geo queries ( )
simplify getInterfaces jtt test
bug fixed for mbean register
Make toolchain . toolchain configurable .
Use all available hosts in REST tests and allow for real master election ( )
Auto - complete : FQ names completion support fix
fix : do not count nop instructions when considering methods for fallback mode printing ( ) ( PR )
[ GR - 24881 ] Invalid option values for RootNodes created in the TruffleLanguage constructor .
[ FLINK - 3675 ] [ yarn ] use classpath paths relative to the application home
Infers type from core annotations ( )
Update low FD count hardcoded hardware Bitmap blacklist .
Refactor : improve some codes not that graceful ( )
NULL XML and JSON values handle fix
Fix limit computation of NIO ByteBuffers obtained via ReadOnlyByteBufferBuf . nioBuffer
Testing findStableLambdaName and allowing ClassInitializationPlugin to be injected in
Fix ObjTest
Optimize DefaultTpsLimiter ( )
merged back the RC branch
[ inlining ] moved InlineEverythingPolicy to inlining . policy
Constraint seek targetGranule within bounds + simplify tests
Enable passing of arguments to the polyglot engine
Execute actions in action schedule immediately .
[ FLINK - 19027 ] [ network ] Only assign exclusive buffers to RecoveredRemoteInputChannels when recovering channel state .
[ fixed ] bug in World . destroyBody . Passed wrong address - > segfault awesomeness . . .
Polish contribution
Remove IndicesOptions bwc serialization layer ( )
currentThread is not exact
Fix broken test ( ordering )
Add URL for JDK11 builds
Navigator stats renderer fix ( respect horizontal scrolling )
Generic table column create UI
Fix issues in Javadocs and exception messages .
Fail a DFS_QUERY_THEN_FETCH request if all shards failed the QUERY phase .
Renamed takeWhile with Func2 parameter to takeWhileWithIndex
Fetch phase when searching might fail when mapping are updated with type missing , closes .
Add error message for missing servlet request param .
improve field data loader to bulk read doc ids
ResultSet logical model : use visual row numbers instead of physical
Remove unused VirtualFrame parameter .
+ fixed internal caching for LocalVariableTableParameterNameDiscoverer
[ fixed ] issue 799 , TextButtonStyle called super constructor with wrong argument order .
[ FLINK - 14926 ] [ state - backend - rocksdb ] ( follow - up ) Avoid exposing handles that will not be closed from RocksDBStateBackend
SQL editor : highlight occurrences of selected word
Remove conn param hints from PG / Generic
ability to add comment to table / table column / view via UI added
add valueEndPosition
Grab Java resources from _proguard . jar instead of _deploy . jar in AndroidBinary .
[ FLINK - 7609 ] [ examples ] Adjust WindowWordCount default window configuration values
Add setCoreFlacExtractorFlags on DefaultExtractory
fix condition for the number of background compilation threads
improve log info
Don ' t close the connection whenever Expect : 100 - continue is missing .
Move negating RequestPredicate to RequestPredicates
[ inlining - 7 ] end of refactoring trail , helper methods now closer to users
Don ' t sort span names in elasticsearch ( )
[ fixed ] this shit http : / / groups . google . com / group / android - developers / browse_thread / thread / 8b14591ead5608a0 #
[ FIXED JENKINS - 16319 ] Stream ordering problem prevented SecretRewriter from working on Windows .
Fixed - don ' t send PING command for blocking queues .
Stats render extra check
Correct the class of Logger in AccountServiceImpl ( )
SkylarkCallbackFunction no longer cares about the obsolete ' cfg ' parameter .
A minor refactoring .
Message . PROPERTIES is a known one
[ GR - 21605 ] Fix getAllStackTraces potential deadlock .
fixed up some left over TODO comments
Adjust BWC for peer recovery retention leases ( )
[ FLINK - 10233 ] [ core ] Reverse deprecation of ConfigOption # withDescription ( String )
Block unknown group messages from blocked senders .
Avoid NPE on null action
Remove - Xclang - only = prefixes .
SQL prefs : set " refresh schema " and " open separate editor " on by default .
Migrate even more tests from TPETest to LPRTest
Put stdout and stderr streams into a consistent state when they are redirected .
Update LoopInitializer . java
MINOR : Fix NPE in KafkaAdminClient . describeUserScramCredentials ( )
Do not compare bitrates of audio tracks with different languages .
Issue 1417 support VEVENT DURATION property
Delete ctx . template_action and ctx . experimental_new_directory
[ GR - 8151 ] Limit thread scaling and promptly shutdown pool .
DefaultPropertyKey private member variable accessed outside scope
Use sendWithTimeout in EventBusBridge so reply handlers are cleaned up if a client does not reply .
Support removal of ranges from ShuffleOrders
Fail fast on getKeyFromServer ( ) failure
Bitly links now support HTTPS - Yay !
Fix bad toXContent for derivative aggregation
[ ML ] allow datafeeds to run if there are any concrete indices ( )
Make Util . inferContentType marginally smarter
[ FLINK - 8099 ] Reduce default restart delay to 1 second
fix : allow cross - block move inline ( )
Add reactive WebSocketClient and RxNetty implementation
Desugar Long . unsignedDivide and unsignedRemainder
Full object name gen fix
fixed NativeFunctionInterfaceTest for locales other than en_US . UTF - 8
this call to launch causes a partially constructed object to escape the constructor .
Tabbed form editor fix ( form initialization )
2 . x fixes
Disable P2 update
[ hotfix ] [ tests ] Remove unnecessary try - catch usage
temporarily disable the confirmation / alerts / prompt tests while the translator ' s difficulties are ironed out
Widen the default - accepting methods of Itera * s
[ logging ] Added an option to pass encoding for log configuration . ( )
2 . x : Fix window ( ) with start / end selector not disposing / cancelling properly ( )
Enhance filter ip address service
see 02 / 26 log
[ grid ] DockerOptions formatting style [ skip ci ]
ERD hints
Fix problem where interpreter calls where not count if the caller got compiled .
KAFKA - 10651 : read offsets directly from checkpoint for uninitialized tasks ( )
for , use UpdateStatement . setAssignment replace of UpdateStatement . assignments
[ JENKINS - 21695 ] - The CompressionFilter . uncaughtExceptionHandler must not attempt to write to a committed response . ( )
datavec python ensure host ( )
[ ML ] Exclude analysis fields with core field names from anomaly results ( )
Fix BEP to report SKIPPED with TargetCompletedId instead of ConfigurationId
upgrade guava to latest version
Remove the - - prune_cpp_modules command line option .
Push more common resources into the handler
KAFKA - 7193 : Use ZooKeeper IP address in streams tests to avoid timeouts ( )
PG database create fix
non - static fields in base version of GraalHotSpotVMConfigVersioned must not be final
[ SPARC ] Throw BailoutException when branch displacements get too large
simplify junit
Remove isFirstWindow / isLastWindow from Timeline .
Fix bug where predictions append to the previous prediction
extract subquery from fromClause
[ tinker ] Try old classloader if class is not found in new classloader .
Proper handle of project lose in DBNModel . + Reopen project on SQL script opening .
Recovery : add a timeout to local mapping change check
[ FIXED JENKINS - 3539 ] Prevent empty file creation if left empty
per SnippetTemplate metrics now include the details of the specialization in their name
[ GR - 23688 ] Search cl . exe version string so that it works for non - English setups .
Log the api url for failed requests
Allow overriding of DefaultDashChunkSource . getNextChunk
Typo .
[ FLINK - 11910 ] [ YARN ] Add customizable yarn application type
experimental UI : improve message shortening
FileProcessor , copy constructor , change processDir to be called even if input files is empty ( simplifies processing in many cases ) .
Removed automatic spilling in SpillingQueue
Fix file navigation typing issues
[ GR - 11687 ] Rename basic configuration to native .
Polishing
sulong : improve error message in sulong launcher if file cannot be loaded
Logic error in what exception to mutate
Honour SSL key and trust store providers when configuring Netty
access type changed
add the ability to control the content type the map will serialize under
Polishing
[ GR - 26184 ] Avoid StackOverflowError during reporting of unhandled exception .
Simplify Shape . hasInstanceProperties ( ) check .
Snapshot / Restore : better handle incorrect chunk_size settings in FS repo ( )
Remove dead code around kept_resources . txt , an artifact which used to be sent to AAPT2 .
ScrollPane , added getWidget .
clean code
Clean up handling of encrypted samples in fmp4 / webm extractors .
SimonStewart : Clean up the MakeProxyPacUnitTest to use guava instead of handling reading a file the hard way
try harder to avoid side - effects that perturb replay compilation testing
Better error messaging when postings_format can be resolved or when a custom postings_format type can ' t be instantiated .
Make TYPE_HIERARCHY_AND_ENCLOSING_CLASSES annotation search strategy defensive
Consider runfiles symlinks when computing Runfiles . getEmptyFilenames ( ) .
[ GR - 7630 ] Support nested service types in ServiceProvider annotation .
[ GR - 19592 ] Revert incorrect handling of uncompressed addresses .
Deps resolve ( order & optional )
Use CustomCommandLine directly instead of via SpawnAction . Builder .
[ GR - 12702 ] MultiLanguageShell $ RuntimeIncompleteSourceException .
MINOR : Serialize the real isolationLevel in FetchRequest
Log a warning if repository autscan is set to default package
[ FLINK - 18552 ] [ tests ] Update migration tests of StatefulJobSavepointMigrationITCase ( Java version ) to cover migration from 1 . 11
Avoid reinitializing data section on each iteration
modify ShardingDatabasesAndTablesConfigurationRange
Core help updated
For , Add EncryptRuleChangedEvent
remove some output
Add method to Requirements to get not met requirements
moving version to 1 . 0 . 0
Added_option_to_open_selected_part fix4
Fix for shift handling
VariableNameSameAsType
added subset toString ( ) for improved rendering
Fix StoreIndexedNode lowering : use a FloatinReadNode instead of Floating a ReadNode
Export binary columns + copy in clipboard : NPE fix
Add start and stop time to cat recovery API ( )
Support registering writable data - source for GatewayFlowRule and customized ApiDefinition ( )
Lint : ' size ( ) = = 0 ' replaceable with ' isEmpty ( ) '
Fix the initial reput offset , according to the maxPhisicalPosInLogicQueue
for : refactor assert AggregationSelectItem
[ hotfix ] [ TE ] Remove redundant HeartbeatManager calls
Fix checkstyle
fixed FileHandle # read
Incorporate feedback from @ jglick
peeling : track times
[ SPARC ] eclipseformat
Exclude password reset api
MINOR : Add explanation for disabling forwarding from value transformers ( )
Install segfault handler only for executables by default
phase contracts : disable code size contract check for expand logic
fix : improve type inference for generics in invoke insn ( )
[ grid ] replace mistaken wording . No logical changes
Improved robustness of lookup protocol
Relational operator IN handle
clean up a bit more
add the delete unused metric config
Stop GenQuery evaluation when an error is found in the transitive closure of the scope .
[ ROCKETMQ - 296 ] fix DefaultMessageStoreTest bug : wait more time for consume queue build ( )
code formatted
[ hotfix ] [ tests ] Remove redundant null check
[ FLINK - 5864 ] [ cep ] Fix duplicate output patterns problem
change routeUnits to final
HLRC : Fix Reindex from remote query logic ( )
Refer to CharMatcher . javaLetterOrDigit ( ) in a comment .
TraceRA : make addInterTraceHints static .
Avoid using IOUtils . readFully in extract - bitcode command .
BAEL - 4488 Convert class with main to UnitTest class
Remove SimpleCache hacks that are no longer used
Now calls setSizeFromLayout ( ) rather than fixed to a 320x240 size
Use castExact for non - final types .
[ hotfix ] [ hs ] Deduplicate variables
Added drawStageBackground method .
Update Nacos SDK version to 0 . 8 and update Nacos namespace demo ( )
Fixed Java docs
[ GR - 4608 ] PE stack traces differ depending on NodeSourcePosition tracking .
Concurrency Fixes for RxRingBuffer & Merge
Makes decoding a very low - pri foreground task but slightly more important than a background task .
Delay loading of a script until the context is ready
Task edit UI fixes
TraceRA : add BiDirectionalTraceBuilder .
don ' t overwrite the values of HotSpotPrintCompilation and HotSpotPrintInlining
Elasticsearch 2 . 4 . 0 uses Lucene 5 . 5 . 2
Use orderedLoopEnds in ControlFlowGraph # connectBlocks .
Worked on broadcast example job
[ FLINK - 12159 ] . Enable YarnMiniCluster integration test under non - secure mode
SHA256 attachment MACs .
closes , inconsistent trailing slashes in IOSFiles
Fix Wrong Value Increment
added Repeat option to CTW
removed JDK6 dependency
[ GR - 18802 ] Ignore missing static libs if - H : + ExitAfterRelocatableFileWrite is set .
for , refactor EncryptDataSource . databaseType ' s class type from DatabaseType to DbType
Restructuring SslHandler to support new engines
Expose some methods from JavaSkylarkApiProvider in JavaInfo
D8 desugar : Handle overlapping bootclasspath and classpath
[ FLINK - 17391 ] [ filesystem ] sink . rolling - policy . time . interval default value should be bigger
core : fix processing ' if ' at loop end
[ TextInputLayout ] Fixing outlined text fields not taking into account top padding for inseting the cutout properly .
Use 2 compilation threads by default , regardless of processor core count .
fixes broken toolbar actions on nodes in the function graph
Make ipa_post_processor documentation show up .
Use opaque in AArch64ArrayAddressTest . useArrayIntLoop to prevent further optimization
Clean up of scheduler strategies . Make sure implicit null checks are only considered in the fixed reads schedule .
Make # getNames return a List , this is more convenient in use and prevents some unnecessary copying
[ GR - 8500 ] Any inspect option starts the server .
Make the simple bitmap releaser its own class
Remove unnecessary parens
Using saved system properties to enable JNI libgraal tracing .
Fix LoggingOutputStream to work on windows ( )
Make the action completion watchdog wait instead of waiting for the first
[ FLINK - 2178 ] [ gelly ] Fixed groupReduceOnNeighbors bug
Add optional user execution time and system execution time fields to TerminationStatus , and also add a TerminationStatus . Builder and tests .
revert by logging error msg . ( )
modify the cat dump channel
[ tinker ] Bugfix : Class XXX has already interned .
Fix small display problem when extension starts with 9
[ GR - 23183 ] Handle try - catch blocks on the same level .
diskCacheFormat - > bitmapCompressFormat
bug fixed for statementExecutableSqlLogEnable
another style fix
Regular getBeanInfo call for interfaces ( enabling Introspector ' s cache )
added parentheses for clarity
Truffle : changed return types of condition profile factory methods .
fixed .
fixed the watcher info API
Mappings : Fix ip mapper to correctly serialize a null null_value
[ FLINK - 11887 ] [ metrics ] Fix latency drift
ERD edit : associations resolve fix
Fix SPARCAllocatorTest junit assumption .
Fix customJodaDateTimeFormat to specify time zone when creating Date
Simplify DynamicHub adjustments for Java 11
Fixes JENKINS - 13822 : expand variables in " Root POM " field of Maven plugin
Add static creation method and early canonicalization for ClassGetHubNode .
Fixed perfomance bug ( repeated image decoding ) . Fire
More robust when Truffle is loaded by multiple independent class loaders
Parameters model refactoring
moved deep array equality comparison into base test class
fix CASE . . . WHEN . . . END indent
Empty objcprovider keys are exposed to skylark as empty sets , instead of throwing an error on access .
Removed duplicate test and renamed another .
ErrorProne ( http : / / errorprone . info / ) suggested fixes for PackageFactory .
[ TEST ] Fix JiraIssueTests . testEquals ( )
Fixed NPE .
Update MapperMethod . java
Truffle : NPE guard in InstrumentationNode
Update to ExoPlayer 2 . 11 . 5
Polishing
Made latency check in unit test more lenient to account for Travis slowness
Tolerate possible null Flyway MigrationVersion when using Flyway 4 . 0
[ TEST ] sync our fork of ClusterDiscoveryConfiguration with core
clean code
Bump to 2 . 5 . 1
Shortcut handling of bodyToFlux ( DataBuffer . class )
SQL completion fix
[ hotfix ] Remove Nonnull annotations from DispatcherFactory and sub classes
Query DSL : Fix ` bool ` parsing .
Fix test failing after f
Log stacktrace
Ignore FrameState when matching nodes
Add test for reusing automatically created engines ; Improve error message .
[ FLINK - 12718 ] [ hive ] allow users to specify hive - site . xml location to configure hive metastore client in HiveCatalog
Rearrange scopes and timers in emitBackend to match up better
[ QUERY ] Separate parsing impl from setter in SearchParseElement
Update tests for RippleUtils . shouldDrawRippleCompat to use parameters .
Clean up of ValueProxy code . Use utility AbstractMergeNode # isPhiAtMerge .
Change file changes listener for resource watcher to an interface
refactor ConfigurationLoader
fix busy loop caused by singledThreadExecutor
validate special case invocations
Actually map Thenable helper interface to JS IThenable . The name is different to avoid a clash with elemental2 IThenable .
refactor parseSelectList 6th version
Fix a regression where writing a ByteBuf are discarded
Do not attempt to load Uris which point directly at a contact photo .
Use separate query ID space for different DNS servers
[ TEST ] Set minimum_master_nodes for new node as well to make sure all ndoes have the same settings
Change Progress indicator to only show if they are already visible
TRegex : unconditionally create ProfilingCharMatchers for complex matchers
use Collections
bugfix : fix NPE in UndoExecutorFactory ( )
Disable flash policy handler by default and use Boolean . getBoolean ( ) to get boolean sys props
[ hotfix ] Make PackagedProgram ( Class < ? > , String . . . ) constructor public
Rander sum as zero if count is zero for stats aggregation ( ) ( )
add static
Polishing
Fix a race in cancellation in DecodeJob .
KAFKA - 6833 ; Producer should await metadata for unknown partitions ( )
[ hotfix ] Fixes broken TopSpeedWindowing scala example
[ hotfix ] [ test ] Wait CompletableFuture complete
[ FLINK - 3197 ] [ core ] Close InputStream in BinaryInputFormat # createStatistics reliably
Improve reporting of test Http1xTest # testSendFileFailsWhenClientClosesConnection
[ GR - 13363 ] Fix ProcessBuilder . redirectErrorStream ( true ) behaviour for native images
PostgreSQL procedures source : show prosrc if there is nothign else .
Polish " Add descriptions for data source pool metrics "
optimize : reduce the db and network request when undoLog or lockKey is empty ( )
Condition applicationContext ! = null is always true ( )
Ensure realtime ` _get ` and ` _termvectors ` don ' t run on the network thread ( )
Register the overlay view as non - obstructing
Refactor AssertThrows to support Throwable
Fixed a potential bug where close future might not be notified
[ Slider ] Fix slider to invalidate after set value
Revert " Extracting binary and profile from capabilities as late as possible . "
NPE fix
[ GR - 7476 ] Support major class file version 54 .
Provide group in ResolvedArtifactLibrary to enable discrimination
Intercept ActivityNotFoundException for ACTION_MANAGE_OVERLAY_PERMISSION
MichaelTamm : increased performance of Select . selectByVisibleText
Parse CLI ' s ` - d ` or ` - - debug ` flag as turning on debug ( elastic / x - pack - elasticsearch )
fix extension adaptive unit test
Start building against Spring Framework 5 . 2 . 12 snapshots
optimize : An optimization about DefaultRemotingParser ( )
made newpipe handle not available like / dislike count
[ FLINK - 15334 ] [ table - api ] Fix physical schema mapping in TableFormatFactoryBase to support define orderless computed column ( )
Update version of TaskInfo header serialization after backport
for , DefaultOptimizeEngine = > TransparentOptimizeEngine
core : add code lines for while loop
aar_import provides jars also via JavaRuntimeJarProvider .
Make ` loader ` Windows compatible
revert batch handle code
add 2 more
temporarily disable TestWaitInPopupWindow - - works beautifully on my machine , but not on the old machine .
improved mysql parser add_index support . for issue
[ FLINK - 4057 ] Checkpoint Metrics
qualifiedColumns
Fix testCleanUpCommitsWhenGlobalCheckpointAdvanced
Fixed the view ' s regex handling of top level items ( issue )
[ hotfix ] Fix the access modifiers in TaskManagerRunner
test : changed some more inline queries from filtered query to bool query
Add additional methods required for intrinsification of indexOf .
[ FLINK - 18468 ] [ tests ] Harden TaskExecutorITCase . testJobReExecutionAfterTaskExecutorTermination ( )
for compatible .
Ensure MacroOptions are processed in order
Improving native - image AArch64 object file support
Respect ML upgrade mode in TrainedModelStatsService ( )
Actually the test and don ' t call start twice on the same server .
lucene 4 : implement createComponents in Analyzers
Fix code formatting
Add SimpleFastInflatedLocking option
fix bug , the usage of ignore ( ) problem ( )
am 54933dc7 : am 871f7ae0 : am 5a27940c : am 1cb0698d : Merge " Improvements to TextInputLayout " into lmp - mr1 - ub - dev
[ GR - 16353 ] Added MBean operations .
Empty list in ids query should not fail , but match no docs
Add threadpool injectability to invalidator
[ GR - 4817 ] AOT : Make sure replacement nodes get valid frame states .
Make proto blacklisting work when virtual import directories are present .
2 . x : Fix Observable . flatMap scalar maxConcurrency overflow ( )
CB - 348 Connection origin info
UriTemplate properly quotes variable values ( SPR - 6854 )
More flexible cipherSuite matching
Add a hint of ownership transfer when calling EmbeddedChannel . read * ( ) methods .
Move the option output formatting into LLVMOptions
Fix compile error for ShardingRule ( )
Fixed Truffle child iterator should only iterate children which are annotated with @ Child or @ Children .
Remove SharedMatchValue
Revert deleted newInstance plugin in StandardGraphBuilderPlugins .
Use NativeLibrary in pointer comparisons .
Cache boolean flags instead of full Annotation objects
Polish " Unwrap InvocationTargetException in isLogConfigurationMessage "
[ FLINK - 4056 ] [ tests ] Harden SavepointITCase
modify AliasExpressionParser
use singleton reduction functions in count and countLong
reduced BeanDefinition footprint by initializing Sets and Maps with 0 ( SPR - 7491 )
Fix BitmapFont parsing ( )
Changing constant interface pattern with a Noninstantiable class
TypeDescriptor equals implementation accepts annotations in any order ( SPR - 9084 )
parse query with offset
[ FLINK - 3339 ] Make ValueState . update ( null ) act as ValueState . clear ( )
[ FLINK - 1220 ] Promote various log statements to INFO level . Add log statements for assignment locations
[ FIXED JENKINS - 14982 ]
more tests
sulong : restrict Mach - O xml parser
Disable streaming when reading to Resources in RestTemplate
Truffle : Fix a few stale references to SlowPath in method names and comments .
Added __syscall intrinsic .
Don ' t count metadata fields towards index . mapping . total_fields . limit ( )
feat : ( add customize resttempalte timeout . )
For , Add LocalShardingRuleCondition
joda time formatting tests
Move ` @ SuppressWarnings ( " unchecked " ) ` on variable declarations out to methods .
Fixes a compilation issue
Fix compiler error
Refactor paging statement test cases for oracle and sqlserver .
Ensure that lazy session id generator is initialized by manager
Add disabled color for expanded and collapsed label .
Update to latest truffle
Fix bug in ContentLengthInputStream where reading a single byte does not adjust readSoFar correctly .
Add hairline border to avatars .
[ fix ] node costs : box node is a normal allocation
make all rest tests pass
modify AggregationDistinctSelectItem ( )
, replace error message
Ensure that Jersey and Jackson honour JAXB annotations
Handle and / or / bsf / bsr without size suffix
Term Facets : Add ` reverse_cout ` and ` reverse_term ` to ` order ` options , closes .
bug fixed for pg sql parser .
Add fillContents assertion to hybrid allocation
[ FLINK - 7225 ] [ core ] Fix exception message in StateDescriptor
[ hotfix ] [ runtime ] Introduce ZERO ResourceSpec to represent zero amount of resources
Improve error message when a persistent worker sends an unparseable response .
add testcase
Forbid negative ` weight ` in Function Score Query ( )
better failure logging when failing to read a tempalte
format update for strict build
assertAddInsertColumnValueWithSet ( )
[ FLINK - 3461 ] [ runtime ] Remove redundant check in ZooKeeperLeaderElectionService
Internal change
refactor ParsingSQLRouter . getGenerateKey
Update Assertions in LLVMForeignCallNode
Fix user copied in SecurityContext . executeAfterRewritingAuthentication ( elastic / x - pack - elasticsearch )
Support configurable Undertow listen addresses
Try to more reliably include http status codes in HttpUrlFetcher exceptions .
OnSubscribe : DelaySubscribe - > UnsafeSubscribe
[ FLINK - 15033 ] Remove unused RemoteStreamEnvironment . executeRemotely ( ) ( FLINK - 11048 )
Fix invalid return value of polyglot_import builtin .
support old registry model and new registry model
integrate SettingsProvider to Maven Builder for free - style projects
Fix racy test
JavaInterop : fix overloaded varargs method caching bug .
modify AggregationDistinctTokenAssert . java
Fix doc - update interceptor for indices with DLS and FLS ( )
Allow default value for CONF_FOLDER to be set at build time
for , decouple ShardingSelectOptimizeEngine & ShardingWhereOptimizeEngine
Additional secure DummySurface device exclusions
- Fixed player new share intent causing main player crash due to player activity in background .
AndreasTolfTolfsen : Return list of available log types in this instance of LocalLogs
Fix SamplerAggregatorTests to have stable and predictable docIds
[ changed ] Sequence , sort of reverted . No fucking idea .
[ FLINK - 15375 ] [ core ] Introduce MemorySize # toHumanReadableString .
Javadoc etc for @ jtnord
Fix compare in LoweringPhase . processBlock
In the SingleTarget case of TargetPattern # parse , store the parsed Label . Then , use this Label during TargetPattern . SingleTarget # eval .
Restrict pipe change for multi - file import
Fix retries if failure occurrs when reading seek map .
[ FLINK - 17842 ] [ network ] Remove NextRecordResponse to improve deserialisation performance
fix lowering of compute object address node
Added simple logcat image perf logger and enabled perf logging for PJPEG sample
fixed ( cluster ) : fixed raft cluster state
implemented resource support .
[ FLINK - 3256 ] Fix colocation group re - instantiation
More comprehensive exception message
Handle records with different structure ( fix error log spam )
Switch to use guava type token . Should be a no - op
SimpleCache : Tweak comments related to blocking
add logic db
[ TEST ] Refactor LegacyGeohashMappingGeoPointTests to 2 . x indices only
MINOR : Fix remaining core , connect and clients tests to pass with Java 11 ( )
[ ] PlatformDependent . freeDirectBuffer ( . . ) now respect hasUnsafe ( )
Plan nodes props fix
Tests : Use all found index files instead of static list for static bwc tests
Columns configuration dialog fix ( restrict exclude of required columns )
reg alloc : do not spil split interval if reg first use is out of range
Support composite registries in MetricsEndpoint
Introduce graal_detach_all_threads_and_tear_down_isolate API function .
Variables in connection properties support
Fixed - Spring Boot default connection timeout is 0 seconds .
[ FLINK - 6353 ] Fix legacy user - state restore from 1 . 2
Stop redirect chasing through filegroup ( ) rules .
Copy Equivalences methods to Equivalence .
disable VTable stub inlining
Polish tests
fixed testcase
[ FLINK - 17330 [ runtime ] Merge cyclic dependent pipelined regions into one region
Handle node id shorter than the usual randomBase64UUID .
Folders coloring
Do not return a non - interop value from TRegexLazyCaptureGroupsRootNode
Oops
Provide definition to allow MachO Objectfile to be created for ARM64 arch
Fix TypeExtractor to work with plain Tuple and derived Tuple
Make HlsMediaSource . Factory take a factory of trackers instead of an instance
[ JENKINS - 34755 ] - Polish the code a bit
Fixed Compiler warnings , closed the scanned and fixed some typos
add setRegistry ( )
[ TEST ] Remove incorrect data frame analytics state assertion ( )
Add AtomikosTransactionManagerRecoveryTest . assertCurrentThreadCannotDoEnlistInDoubtState ( ) .
Fixed unit test
Use Windows - specific custom features so download URI is correct
fixed style issue
Flip the - - incompatible_depset_is_not_iterable flag
Remove redundant stuff
fix : fixed future . join blocking thread
bug fix for StateAnalyzer NPE
Fix InviteActivity sendSmsMessage type error .
fix numerical issue in function score query
Fix md2 and cache for UTF16 indexOf on Solaris .
[ tinker ] Bugfix .
[ GR - 9258 ] < launcher > - - inspect does nothing .
Fix a crash where dataSetIndex is - 1
Folder switch fix
[ FLINK - 13012 ] [ hive ] Handle default partition name of Hive table
[ GR - 26368 ] Throw proper exception for truncated binaries .
Fix typo
Mistyped test name .
Adjust available and free bytes to be non - negative on huge FSes
added a ToXContent implementation to NodeStats
refactor DataSourceConnectionUrlUtil ( )
[ GR - 23366 ] Instanceof of dynamic : properly route exact flag to snippets .
Fix compilation warning when leaving the context on SVM .
Changing the instruments cache to a WeakHashMap with ClassLoader as a key
[ FLINK - 10149 ] [ mesos ] Simplify assertions in LaunchableMesosWorkerTest
Use List instead of priority queue for stable sorting in bucket sort aggregator ( )
[ FLINK - 6776 ] [ runtime ] Use skip instead of seek for small forward repositioning in DFS streams
Fix comment typo
Update Update8051 . java
SQL console title - include object name
byId needs to be copied , too
Allows IP_TRANSPARENT to be set on a redirecting socket
Added support for generic interface Animatable in placeholder drawables
add antlr visitor for MySQL TCLStatement . ( )
De - flaking PageLoadingTest . Tests are stable over 100 runs .
[ fixed ] Fixed issue 497 .
minor changes to AssetManager
[ GR - 25728 ] OptimizedAssumption # invalidate loses the invalidation reason and always uses " assumption invalidated " .
Treat the presence of a multiple attribute as a sign that the select supports multiple selections .
Fix Profiler setup to actually enable the Json format
Add cue positioning to SubtitleWebView
Reduced logging during task recovery
fixed code violations
Standalone native image should parse native / jvm options
Register prefixed path patterns with FixedVersionStrategy
dbeaver / dbeaver - ee enhancements
[ runtime ] Miscellaneous style cleanups
Add @ NonNull and @ Nullable annotations
Add post - migration coverage for testObjcProtoAspectPropagatesProtobufProvider ( )
Release SearchContext releaseables also post match filters / queries .
[ FLINK - 7197 ] [ gelly ] Missing call to GraphAlgorithmWrappingBase # canMergeConfigurationWith ( )
Truffle - DSL : use clone ( ) for copying node arrays in copy constructor .
Warnings .
Made Payload available on dragstop without target
Prevent Leaking Search Tasks on Exceptions in FetchSearchPhase and DfsQueryPhase ( )
BAEL - 12 userAgent header value in one place
improve sql parser
Update group size flag .
PG view DDL + view edit
allow null for @ ConstantParameter values
Rename thinlto_param_file to thinlto_indexing_param_file for indexing actions
Add UnsafeMemoryStoreNode and use it for Unsafe . put * ( address )
Boolean compare ordering switched
query_string on multiple fields with " * " fails in 0 . 19 . 9
[ PR - 446 ] Add FastR sample to the tutorial and TCK tests .
Emit settings deprecation logging on empty update
add more debug statements to BasePlayer
Finished implementation of buffer utilization log
Update value view panel on cell re - select
Make DownloadThread extend Thread
Global cleanup change .
[ FIXED JENKINS - 23893 ] Top level ListView is not updated when job is renamed / deleted
Avoid truffle compiler initialization in createCallTarget and cancelInstalledTask .
Fix scrolling in the emoji variation popup .
Add ClusterSearchShardsAction to " read_cross_cluster " privilege ( elastic / x - pack - elasticsearch )
Improve ` arc ` geo - distance accuracy .
[ fix ] RegisterAllocationTimeBenchmark should use TraceRA if requested
MichaelTamm : fixed issue 708 as suggested by Michael Vorburger
Handle core ' s cleanup of terms aggregation
[ grid ] Making code Java 8 compatible
Reduce raw stacktrace dumping
RSV filter dialog UI glitch fix
Support pointers to managed ( foreign ) objects as return buffer in getcwd syscall .
[ FIXED HUDSON - 2097 ] remote API of Computer was not working because SwapMemoryMonitor wasn ' t producing remotable data .
Thread safe for DatetimeShardingAlgorithm ( )
Cleanup
Refactor using executeSQL method .
Try to prevent unrolling nested sets in NestedSetBuilder .
Audio focus : Re - request audio focus if in a transient loss state
ShardUtils # getElasticsearchLeafReader ( ) should use FilterLeafReader # getDelegate ( ) instead of FilterLeafReader # unwrap
[ fix ] mitigate OutOfMemoryException in linear scan
convert object to string before set properties to SPI instance ( )
[ GR - 22472 ] Fix AArch64 AdrpLdr patching mistake .
Wire up GeoDistanceAggregation ( )
RxJavaPlugins performance enabling fix & lockdown .
[ changed ] using the fixturedef in the box2dtest now .
Fix support for snippets that use a receiver .
[ GR - 16672 ] Implement IOUtil . write1 ( ) .
Annotations ( explain plan )
Include import for RootNode when the handler of IS_POINTER message is not defined
Enabling . htm and . xhtml extensions for test suites . Fixes issue 6496
[ FLINK - 9275 ] [ streaming ] Add taskName to the output flusher thread ' s name
Fixed issue 531 by removing unnecessary calls to set compound drawables
Add active status to group multi - device sync messages
for , ShardingOrPredicateFiller . fillShardingConditions = > createOrCondition
RSV : SQL script gen - columns order fix
Ensure - H : Path gets canonicalized
[ inlining - 4 ] privatizing methods that can be made private
Migrate Cast demo app to use DefaultDrmSessionManager . Builder
Animation improvements
Fix execute REST API call
Better formatting
Avoid default size for small collections .
for , remove ParseRuleDefinition
[ GR - 6571 ] New instance of java . lang . Object cannot be created using Truffle Java interop .
HttpRange validates requested ranges
add clear when deleting an index for the field data service
Modernize LoginRedirectTest
URL Cleanup - fix undesirable code change
Adds flag to shorten header output directory root from " _j2objc " to " _ios " ,
test : increased logging for http and improved assertion failure
Make newArray stubs reexecutable .
Fix import
provide MessageIterator instead of Iterable
Adds ElasticsearchSpanStore . MAX_RAW_SPANS to permit large query results
[ guava concurrent ] Inlined getDone ( ) and getUninterruptibly ( ) to break the source dependency from AbstractFuture to other classes in / util / concurrent .
reverted temporary commenting out of MonitorTests
avoid unexpected automapping when a explicit mapped proerty has the same name as another column
Defensively use setRemoveOnCancelPolicy for JDK 6 compatibility
Write some tests prescribed by missing code coverage . I found a bug where our nonexecute prefix code causes a problem .
Check for degenerated lines when calculating the centroid ( )
Use factoryBeanObjectType attribute to find factory bean to replace
New execution plan style ( output )
convert arguments to pointer
[ FLINK - 3423 ] [ tests ] Fixed ExternalProcessRunnerTest on Windows
Fix NPE in AbstractMessageConverterMethodProcessor
TEST : Rewrite GeoPointParsingTests # testEqualsHashCodeContract ( )
Fix line length
Optimize path / text evaluation in ninja lexer by using equals instead of set lookup
Remove deprecated members in DefaultTrackSelector
Command cntext : commit only in txn mode
Adding more examples to the Quartz project ( )
Added readField method that can be used for a static field .
[ minor ] Refactor getJarFiles ( ) out of ScalaShellEnvironments
Transfer status fix
Add bytecode interpreter partial evaluation test with loop .
Some formatting changes , and a few tiny optimizations
Truffle : made TruffleRuntime # getCallTargets return a Collection and simplified its implementation .
Fixed signature of test
Delay Http2ConnectionPrefaceAndSettingsFrameWrittenEvent by one EventLoop tick when using the Http2FrameCodec ( )
* Made sure ServerBootstrap accepts only ServerChannelFactory
Simplest concat test from 1 . x
BAEL - 7 : Use Collection interface
[ JENKINS - 59580 ] Fix WindowsOSProcess . getEnvironmentVariables returning null
delete todo
@ fillUrl should take the public static fromStapler method into account .
Remove defunct Android incremental dexing flags
FrancoisReynaud : adding a timeout to the matcher thread for new request .
Fix Race in Concurrent Snapshot Delete and Create ( )
Increase ResourceShrinker MethodVisitor level to ASM5 from ASM4 so that it can process class jars generated with javacopts = [ " - parameters " ] .
Remove references to Downloads from IntentUtil
Polish VcapApplicationListenerTests
Clarify why we ' re logging a specific failure .
Rookie mistake : only include the w3c capabilities in the w3c capabilities
A setting to auto expand the number of replicas of an index ( based on data nodes ) , closes .
improved date parse . issue
minor modifications
EranMes : Per Simon ' s suggestion , using the PortProber .
Disable by default plugins isolation
fixed issue2355 and added associated testcases
sulong / toolchain : use - save - temps workaround on darwin
security : read correct file when listing users
Catch resize exception and inform corresponding cb
add logging if file does not exist
Fixed deserialization problem in AbstractDeserializer
Reformat the GridLauncherV3
Polish
see 03 / 18 log
Strengthen typing in sorted differences ( )
Marvel : Client nodes should be able to send metrics
use RestTemplateFactory to create rest template , which will respect the api . connectTimeout and api . readTimeout settings in PortalConfig
[ ] remove redundant codes .
Typa and AvroSchema are supplied to AvroJacksonCodec
Remove Matchers class
Correctly set the stack pointer after abstract interpretation of if / loop .
Increase number of forks to 3 for microbenchmarks
Tweak compiler assertion
Add more verbose matchers / asserts to tests
Test : extra debugging for refresh listeners stats
Use constants for props names
[ hotfix ] Harden SlotPoolRcpTest # testCancelSlotAllocationWithoutResourceManager
Support delete service in AP mode
Generalised the name of the expected InputSplit class in HadoopDataSource .
fixed testcase
polish 6322374d81d9e1374c65f5594cf784c37a459fe9
Migrate SessionAutoConfigurationMongoTests to Testcontainers
Fix incorrect template name in test case
Fix bad voice note duration and listener breakage .
ThreadPool . Info and SizeValue to implement Writeable rather than Streamable
Rebase onto the latest dev , update appcompat dependencies to use androidx
Fix release tests ( )
increase teh number of " med " connections from 4 to 6
[ TEST ] Fixed bounds calculation for extended bounds in histogram agg empty buckets test
Fix range tests
Implemented ReplayingDecoderBuffer . toString ( ) properly
[ FLINK - 15065 ] [ docs ] Correct default value of RocksDB options in documentation
Make the access of raw memory via Unsafe specific to HotSpot .
Remove an unused method from CppCompileAction .
Create a new h2 stream only if the parent channel is still active ( )
[ GR - 24281 ] JDK - 8247832 : Optimizing VolatileReadNode out causes wrong memory state to propagate to uses of the VolatileReadNode .
Ensure test replicas have valid recovery state
Polish " Fix Spring Data Cassandra auto - configure condition "
KAFKA - 9509 : Increase timeout when consuming records to fix flaky test in MM2 ( )
security : mention comma - separated for IP and DNS name prompts
fixed broken build . . . . . . . .
Use proper setters when copying FirefoxOptions
Cleanup the code integrated in
, fix for binary protocol value
[ hotfix ] [ table ] Changed ApiExpressionVisitor to abstract class
Output logger flush
Simulating window . getPosition and setPosition operations
Fixed bug when non percolator docs end up in the search hits .
[ FLINK - 14344 ] [ checkpointing ] MasterHooks class supports asynchronous triggering
Make header decorations play well together
Use PackageManager # checkPermission to check for network state .
* : * query string query should resolve to match_all query , since the wildcard field support , it no longer does
Various notification code improvements
Precompute defaults package content to save ~ milliseconds every time BlazeRuntime # getDefaultsPackageContent is called .
One too many . . .
Fix some special cases with other number of DwTags as expected
Add the thread name as metadata to the Json profile
Remove non - resizable option and rename methods
, for checkstyle
fixed createEntityWithRandomPk : create collision free PKs
Renamed vertx - options arg to just options
make internal properties package private ( )
Synchronize when initializing LLVMCallNode # prepareArgumentNodes
fix : StringUtils . split ( String , char ) not correct for input " d , 1 , 2 , 4 " , ' a ' ( )
Fallback image build must not have ` . ` on classpath
Finally , much better support for auto - focus , other UI fixes . Now requires JSR - 234 .
Fix Checkstyle violations
[ GR - 19335 ] Small test fixes .
fix mistyped word ( )
[ TEST ] Temporarily don ' t corrupt . del files since they are generational and we might pick the wrong one
[ TEST ] take into account when master node is stopped
[ hotfix ] [ table ] Removed unnecessary casts in ApiExpressionUtils
Fix a bug in AioSocketChannel where recursive doBeginRead ( ) is allowed unexpectedly
Checkstyle .
fix ( gui ) : limit the spare memory to max . 512MiB ( )
Seek duration should not be rounded up when opening the settings , even if inexact seek is disabled
Upgrade to the Spring Data Dijkstra release train
TEST : Ensure a manual flush always occur ( )
fixed a fragile test
HttpObjectAggregator adds ' Connection : close ' header if necessary
Clear supplementalData in DecoderInputBuffer . clear
[ Transform ] make state handling more robust when stop is called while indexer shuts down ( )
Clarify that the given projection fov is for Y
Fix documentation for android_sdk_repository . build_tools_version .
Have more helpful error messages .
Avoid array allocation in HashCode . writeToBytes ( ) .
Fix missing parenthesis in commit timeout message
For better handling of some formats , perform rotation with help of Graphics2D
highlighter : Save unwrap BytesRefHash . MaxBytesLengthExceededException
don ' t make util depend on core etc .
Don ' t produce multiple calls to exceptionCaught ( . . . ) on SSL handshake failure ( )
Spreadsheet focus handling fix
SQL Server : context default init fix ( active schema for isolated connection )
Handle lower - case values for enums nicely
Update after review
Use Cleaner non - reflectively for runtime - compiled method metadata .
modify the cat producer
Added some clarity to operator precedence .
PEA : fix virtualization of CheckCastNode ( check type )
TextAppearance typeface bug fix & refactoring of getFontAsync ( ) .
BAEL - 175 - Integrtion testing minor changes
add test case
add matrix report
[ FIXED JENKINS - 23027 ] Added Launcher . ProcStarter . quiet option .
Work around issue with Xiaomi JB devices
[ FLINK - 19427 ] [ FLINK - 19489 ] [ tests ] Fix test conditions for ' SplitFetcherTest . testNotifiesWhenGoingIdleConcurrent ( ) '
SPR - 6941 : part ( 1 ) correct exception handling when null cachedExecutor
Update parseStbl to handle edit lists & gapless info
unit - api examples ( )
refactor and add unit test for HeaderPacket
LLVMAccessGlobalVariableStorageNode is a now an abstract class and now takes a context to access the global storage .
Use an interface to convert from nanos to millis since epoch
* Replaced Cookie . setPortList ( int [ ] ) with Cookie . setPortList ( int . . . )
remove _create from POST to match PUT path
For , implement getImportedKeys
Fix failing NettyReactiveWebServerFactoryTests
Loaded sources are not added to executed sources . ( GR - 27071 )
Error in Option
Test fix for cert names ( )
read settings ( json , yml ) using utf8
[ FLINK - 9339 ] [ REST ] Register subtask accumulator handler under correct URL
Fix Rounding test error
Make type wrapping optional for PUT Mapping API request
[ FLINK - 13307 ] [ tests ] Fix SourceStreamTaskTest test instability .
[ TEST ] Use a large threshold to prevent relocations in RecoveryBackwardsCompatibilityTests
Allow custom instances of WebJarAssetLocator
added test for checkcasts inserted by lowering of storeindexed instructions
test : limit auto check execution time
Make it easier to update extractor test dump files
Narrow type of " configuration " field in ActionOwner . It is only ever a BuildConfiguration .
fix compile errors
Refactor : Change the position of variable .
Fix getInstanceFields for arryas and interfaces
[ Cpp ] Allow . so files to have more extensions .
JasonLeyba on behalf of DouniaBerrada : Android cookie handling clean - up .
[ changed ] Stage . render ( ) is now called draw ( )
Transport Client : Improve remote node freeze handling by adding another timeout layer , closes .
Show notification whilst ads are playing but hide playback actions .
Checkstyle .
Better Exception in NetworkUtilsTests ( )
Change of API .
modify fillInsertValuesToken ( )
Allow pluginManager to be queried by superclasses
304 Not Modified overrides Content - Encoding and breaks cache ( )
[ JENKINS - 21984 ] Checking stable UC as well . Cannot reproduce certificate problem in test .
Grouping panel message update
improved error handling
add ability to use RoundRobinDnsAddressResolverGroup in Connection manager .
Automated rollback of commit 52c4d05d1bfd54ea901e9926be7b262dd45e34de .
tests for custom conversion service / validator
Relax the check in SpannedSubject . hasBoldItalicSpanBetween
Access flags , refactor
Fix a couple of errors related to registersComponents refactor .
If custom transformation fails , do not invoke stats .
enhance the checking for sql execution
Make start / stop metods package protected
Release : Add 1 . 7 . 2 bwc indices / versions
Fix for - Cannot disable audio focus after enabled .
[ SPARC ] use now a dedicated scratch register for isMethodHandleReturnOffset value , as o7 may contain valuable information
Fix reporting of total indexing stats
[ GR - 19785 ] Replacing native image system class loader .
Remove dependency on the alert history indices being started in the actions .
Fixed - RedisTimeoutException is thrown if retryAttempts set to 0
Avoid creating Uris when loading Files .
[ FLINK - 2814 ] [ optimizer ] DualInputPlanNode cannot be cast to SingleInputPlanNode
Unit tests method names and class names modified as per the guidelines
ThreadPoolStats to implement Writeable rather than Streamable
[ hotfix ] Properly stop Flink Yarn application in YARNSessionFIFOITCase # testJavaAPI
Test : ensure a no action entry has been fired
Decode attributes which may contain either enums or values , e . g . layout_width
Never release store using CancellableThreads ( )
[ hotfix ] Refactor YARNFileReplicationITCase to avoid inheritance
Adapted MockEnvironment to match new internal structure of RuntimeEnvironments
Removed blocking code from HttpServerImpl . close ( )
[ TEST ] Fix test issues with branches still using mvel as default scripting language
[ scene2d ] Allow node height to be overridden to create a blank space in the tree .
Add retry mechanism to get snapshot method
[ fixed ] issue 453 , thanks tescott .
Fixed to match backend .
Expose cue settings parser
allow to set specific routing index_routing and search_routing names as well
Relax ServletContext check for resource handling
Fix log parameters
[ GR - 10640 ] Handle Windows new lines in tests .
refactor test cases
keyword " ILIKE " to Snowflake SQL Dialect added
test fixed
TraceRA : remove interval dumper and make IntervalData dumpable
FK edit page fix ( update buttons state )
refactor
Tabs close check fixed
Check template availability in ScriptTemplateView
[ TEST ] [ ML ] Run ensureStableCluster ( . . . ) on all nodes and run synched flush before executing the disrupt
Add NettyServerCustomizer beans to the auto - configured factory
allow derived class to control SCM completely by getScm ( ) .
Delete state in the JobStorageDeletionTask ( elastic / x - pack - elasticsearch )
Pasting should respect the onlyFontChars setting .
Improve MIME type subtype suffix handling
Move BlazeRuntime . getView ( ) to CommandEnvironment . Update all callers .
Add DownloadAction merge tests
Truffle - DSL : fixed bug when using @ CreateCast with children arrays .
[ FLINK - 17822 ] [ MemMan ] Use private Reference . tryHandlePending | waitForReferenceProcessing to trigger GC cleaner
sql parser support > =
LongMap , fixed issue 779 , shift was not 64 bit .
avoid string concatenation in FrameDescriptor # getNotInFrameAssumption
[ TEST ] Allow higher ClusterHealthTimeout on tests if configuration is slow
[ scene2d ] Improved DragListener , allow getting drag start position and drag distance .
Pass through script params in scripted metric agg ( )
use local registers object instead of global one
Fix wrong synchronization
Fixed when specializing nodes must always call the full generic case .
Make the Spring Security part of CF autoconfig conditional
UsageStatisticsTest failure : java . vendor vs . java . vm . vendor
Make DeepUnrollingTest more robust
fix typo
Select COUNT from SQL query fix ( remove orderings )
StyledText find / replace
modify parameters
Replace equals ( ) expressions with Objects # equals ( ) where possible
Workaround a cipher issue in Android 4 . 3
fix file format , add final to params
@ is an unsafe character therefore do not use in a project name
[ ML ] Before selecting an executor node for running job task , make sure that all required indices exist and are accessible .
Truffle : simplify NodeUtil . findChildField
Fix aggregation tests
HotSpotResolvedJavaField : don ' t embed caches of java . lang . { Integer , Long , Boolean } for replacements when compiled in AOT mode ( GRAAL - 290 )
Safety number string fixes
Update javadoc of Node # getSourceSection with new examples . Clarify that getSourceSection is not to be invoked on the fast - path .
use Kind . Void for ReturnNode , not the return value kind
remove DHE cipher from default list
Make HttpConnectorTest for timeout less brittle .
Parse build . time as an ISO 8601 instant
polishing
fix for excess buffer size increase bug
do not update mapping if it is newer than the current version ( elastic / x - pack - elasticsearch )
improve code quality
SonicAudioProcessor : Fix scaling to account for pending input bytes
[ TEST ] Don ' t randomize preference PRIMARY it might not try replicas depending on the clusterstate
[ GR - 20990 ] Adds substitutions for Package class in order to handle
Stream version nit after ff0f540 and ce60585
Intrinsify Preconditions . checkIndex to length > = 0 and index | < | length guards .
Add unit test to show that the callback method is called .
[ GR - 13568 ] Parameter " end " of Debugger . getPossibleBreakpoints command is optional .
Fix checkstyle violations
Make Painless the default scripting language .
Fix overflow guard condition for down counted loops
fix compile
MINOR : remove streams config params from producer / consumer configs
Use getAllocationByteCount ( ) for API 19 +
Post cherry - pick fixes .
[ hotfix ] [ kafka ] Fix RackAwareMode instantiation in Kafka 0 . 10 tests
DB2 connection page
Fix style in function call example .
[ inliner ] added a factory method in InlineInfo to make code uniform elsewhere
Add profiler instrumentation to RemoteSpawnRunner
Snapshot / Restore : fix compilation issue for JDK 1 . 7
Data transfer : attribute bindings processing redesigned
Remove Locale . getDefault ( ) call from a final static field as it can change .
Fix intermittent failure with ZK cluster manager
Fix formatting .
Use Builder in ImaAdsLoader constructor
Blacklist OMX . SEC . mp3 . dec and OMX . brcm . audio . mp3 . decoder on Samsung devices
modify the cat topology item build
Restore changes from 0332501 ( fix class loader test failure )
SQL syntax highlight fix ( quoted identifiers )
Instrumentation testing for UnexpectedResult exception handling .
Sanitize sun . java . command by default
[ TEST ] Remove wrong transportName from MockTcpTransport # ctor
Fix compile due to change in InternalSearchResponse signature
[ ML ] add supported types to no fields error message ( )
Move CachedContentIndex storage behind an interface
Grid vertical scrolling
Make it possible for sub - classes to have a sub - class specific trust anchor chain
Ability to disable trigger
implemented table layouter .
Changed es version from string to class Version
use getPeriodByUid when searching for subsequent period of seek timeline
Assume that those path which end in a DeoptimizeNode are taken less frequently .
Store NativeDatagramPacketArray directly in the EpollEventLoop
copy servers before closing to avoid comod exception
Make sure evicted sessions clean up state .
core : remove ' static ' modifier for inner interfaces
Rewrite logic for enabling secure DummySurface
SQLPacketsBackendHandler = > NettyBackendHandler
Clean up player event handling
Allow to customize Casssandra ' s Cluster
[ FLINK - 9274 ] [ kafka ] Add thread name for partition discovery
Make the logger static final
Replace iterable foreach loops with indexed for loops
[ FLINK - 15942 ] [ tests ] Simplify ResourceProfileTest tests
bazel syntax : rename PrettyPrinterTest to NodePrinterTest
remote : Delete output files created by failed download .
SSLEngine . beginHandshake ( ) can throw a RuntimeException .
Fixed AD leaked thread bug
Fix NPE when calling NettyHeadersAdapter . add ( )
refactor ScriptRouter : ( )
[ GR - 7480 ] LoopScope . initialCreatedNodes is a mutable array , so every new scope needs its own copy of the array .
Moved tests to FromCallableTest from FromCompletableTest ( )
Added test base for new java api programs
More than words : removing the message in favor of the actual formula
[ Test ] IndexLifecycleActionTests - set minimum master node earlier to make sure it ' s in effect when needed .
Fix always defaulting to DiskCacheAdapter
Clean up AbstractUnsafe # bind and AbstractUnsafe # disconnect .
Swap header check in ReadOnlyHttp2Headers
Applying SSL cert patch for Issue 3011 ( thanks to Ivan De Marino and Mark Watson @ Neustar )
Use unique ports per test worker ( )
Adjust to xcontent being a separate lib ( elastic / x - pack - elasticsearch )
Add tests
GlyphLayout , fixed [ [ not being escaped when followed by color name .
Add santoni and mido to codecNeedsSetOutputSurfaceWorkaround
GT - 2887 : Fixing iOS DYLD file system disk space issue .
[ SPARC ] Use Fzero ( d | s ) for clearing a fp register
[ FLINK - 2604 ] [ runtime ] Adds synchronization to stop method of ZooKeeperLeaderElectionService
Use messageFont for drawing messageText .
Simplify calculation in AwarenessAllocationDecider ( )
Log more details if notification of promise fails in PromiseNotifier and AbstractChannelHandlerContext
Add flag - - incompatible_blacklisted_protos_requires_proto_info
Fixed wrapping when none of a run fits on the current line .
Postgre Inheritance DDL fix
Allow use of JsonInclude . Value
[ FLINK - 13060 ] [ coordination ] Respect restart constraints in new RegionFailover strategy
Remove HTTP max content length leniency ( )
* JdkLogger passes null method name now - it ' s allowed in the specification explicitly .
fix the primitive case of instanceof
[ ML ] Make GetJobStats work with arbitrary wildcards and groups ( )
support q to quit . close
KAFKA - 8341 . Retry Consumer group operation for NOT_COORDINATOR error ( )
Add configuration map support 15th .
testRecoveryAfterPrimaryPromotion : seqNo recovery doesn ' t require some initial indexing
Fix GeoPointFieldMapperTests expectations .
Use DexBuilder tool such that it can be a persistent worker
Disable failing test in UDP broadcast
Fix the bug that entry in SentinelReactorSubscriber did not carry expected entry type
modified to work correctly on slaves .
Adjust CombinedDeletionPolicy for multiple commits ( )
[ hotfix ] Let TestingOnCompletionActions . getJobMasterFailedFuture return jobMasterFailedFuture
[ FLINK - 15941 ] [ format - avro ] Create schema coder only once in RegistryAvroSerializationSchema
Fix integer bitrate division
[ FLINK - 19162 ] [ connectors ] Add ' recycle ( ) ' to the RecordsWithSplitIds to support reuse of heavy objects .
Fix deprecation warning in SpringPhysicalNamingStrategyTests
[ grid ] Adding router role to Hub , this enables the flags on the CLI
fix ( gui ) : fill background before draw line numbers ( )
Fix cyclic ImageSingletons init for Java 11 +
Fix style issue and incorrect comment
Pass app context to the IMA SDK
Don ' t use disabled background if there is none .
GT - 3089 Accepting pull request , fixed a few more relocations and added a
Adapt BWC conditions after backport of
Let GLVersions be available at create , fixes
[ FLINK - 14205 ] Return DuplicateJobSubmissionException if submitted job is a duplicate
[ FLINK - 11334 ] Remove old deserialization logic from new EnumSerializer Snapshot
Refactor channel
Add - cp to option help for compiler commands
Overload selection : Prefer List over array type if both are applicable .
[ GR - 25055 ] Sulong : read up until the aligned size of a polyglot i8 array
Avoid calling setDataSource twice for videos
Fix Jack ' s proguard specs .
Fix parameter order in SctpOutboundByteStreamHandler .
Fixed
Issue 508
Changed SpatialRelation contains to within
polish ClassUtils ( )
add some tests .
[ GR - 20680 ] Amend ResourcesFeature and NativeImage to support java . xml .
Wrong url after renaming job
1 . 2 . 48
Accept the same boolean values in RestRequest
Clean ` WakeLockManager . updateWakeLock ` logic .
for , refactor VisitorUtils . visitFirstOrAfterColumn
[ TEST ] Don ' t rely on ` new Date ( ) `
KAFKA - 10713 : Stricter protocol parsing in hostnames ( )
RSetCache few improvements
Use main_class as specified in java_test even if use_testrunner is true .
MINOR : Include partition when logging fetch errors ( )
Issue 1061 URL - escape { CODE } replacement
Extract methods for Members constructor .
addEdge fix
Corrected the test .
Navigator double click behavior fix ( leaf nodes )
[ GR - 3627 ] Correct various errors in code generation ( ) .
Minor fixes
SPR - 8174 : varargs and primitive handling in SpEL
Change group calling feature flag to boolean .
Truffle - DSL : fixed an NPE if invalid TypeData are compared .
Constraint seeks within bounds for ConstantBitrateSeeker
Fix potential NPE in ` _cat / shards / ` with partial CommonStats ( )
Removing ' final ' accessor from constructor ( )
NPE fixed ( " no active project " case )
False positive avoidance
Add parameter names to Format creation
[ FLINK - 3313 ] [ kafka ] Fix TypeInformationSerializationSchema usage in LegacyFetcher
Ensure that the TraceInlining option is always checked when producing an UpdateScope .
Create API key with TransportBulkAction directly ( )
Add Javadoc for HystrixMetrics
Removing indices privileges from the reporting_user ( elastic / x - pack - elasticsearch )
Use ArrayDeque for playback parameters checkpoints
Print details for fatal exceptions at a CEntryPoint boundary .
Add instanceof checks before calling earlyCanonicalization
Fix InstallPluginCommandTests failures on windows build
Fix reput problem
Auto - recover after data reading
Amend ` prepareIndexIfNeededThenExecute ` for token refresh ( )
[ FLINK - 13284 ] [ table - planner - blink ] Correct some builtin functions ' return type inference in Blink planner
for , refactor DatabasePacketCodecEngine from SPI to DatabaseFrontendEngineFactory . getCodecEngine ( )
Watcher : Do not pause watcher if not needed ( elastic / x - pack - elasticsearch )
Fix the incorrect usage / value of ' Connection : upgrade '
Reindex : Better error message for pipeline in wrong place ( )
Improve performance of StringUtils . capitalize ( )
SimonStewart : Amending the Remote Client to allow the executor to be switched . There is a build failure . Fix will follow
Added testPluginActive to catch problems like in JENKINS - 28816 .
update , move deferredresult controller ( )
compilation fixed
fix WORLD_MODE_READABLE bug
Protect ScriptedMetricIT test cases against failures on 0 - doc shards ( ) ( )
fix compile errors after optimize api renaming in core
Fixed LiveTests for migration of spring - boot module to Spring Boot 2
Fix soft body face culling
SQL query parser error handling fix
chore : fix indent of first line in fallback mode
FB annotations in TimerTrigger . class
[ FLINK - 11759 ] Update AbstractNonKeyedOperatorRestoreTestBase for 1 . 8
Cleanup runtime options cache .
Make ` native - image - cp ` tolerate non - existent classpath entries
[ FLINK - 20429 ] [ kafka ] Using proper watermark interval in KafkaTableITCase # testKafkaTemporalJoinChangelog test
bug fixed for LogFilter
[ GR - 4534 ] Temporarily disable Truffle OptimizedCallTarget entry point release on AArch64 .
Generic model fix ( generic types )
updating non dynamic settings throws an error now
Password lookup ui fix
modify system monitor nag order
Changed the behavior of CompilerDirectives . injectBranchProbability and added javadoc to document the new behavior . Introduced probability constants . Removed CompilerDirectives . slowpath ( ) .
test : refresh only once to workaround phrase suggester edge case
dont spam ( )
Update stale comment in TrimmingAudioProcessor
Fixed issue where self - sends didn ' t auto download .
ConstructorResolver exposes parameter signature from user - declared class ( in case of a CGLIB - generated subclass )
Reduce contention in the digest map .
Polishing
Support DDL for Oracle visitor ( )
Rewrite ServiceManager to use MultimapBuilder
Put back AWT for getting DPI .
Content viewer actions
Automated rollback of commit 87f19705026e90e774ce6773299429b8267ecfe8 .
Remove Google specific headers from HttpHeaders .
modify unit tests
BAEL - 3961 move the sql string into a new line to improve readability .
mapping work
reflection workaround for JDK - 8213406 changes
Updated CombSort . java
Make manual multidex incompatible with . dex sharding .
Changed method name in Spring Aspect .
Polish
flush with full = true is broken : this IndexWriter is closed , closes .
Marvel : Add indices options to Index Stats / Recovery collectors
More test cases : Round one
Removed BuildConfiguration . Fragment # getName ( ) and # isSkylarkVisible ( ) .
[ TEST ] Wait for yellow since some shards might not be started
Bump version + update release notes
Fix instrumentation root node bits handling .
Change the UUID of actions consuming the set of build info keys .
timeout increased for jdk 10
PostgreSQL 10 + sequence show in tree fix
Work around DSL bug by not sharing nested library .
[ hotfix ] Fix checkstyle violation in TaskSlotTableTest
handle @ argfiles properly on JDK 9 + command line
Rename DiscoveryNodes # smallestNonClientNodeVersion to getSmallestNonClientNodeVersion
Connections aren ' t closing during RedisClient shutdown
Update BuildIntegrationTestCase to use registry methods instead of executorInit where possible .
Reducing visibility of DriverService . sendOutputTo method to protected ( this change was announced in version 3 . 8 )
modify sharding - ui - backend module for orchestration . ( )
BitmapFontCache setColor changed to match SpriteBatch / SpriteCache .
Correct some unit test cases
BugFix : AsyncSubject
Fixed - Redisson Tomcat Manager doesn ' t store principal and authType session attributes .
minor change
Skip incomplete mysql rows instead of NPE
Fix comment typo
add logging on which state file failed to load
issue fix the magic number first
Fix two rare crashes from bad input / state
KAFKA - 10028 : Minor fixes to describeFeatures and updateFeatures apis ( )
Ensure AnalysisMethod . getParameters ( ) never returns null
Reset Truffle compiler after running all encoded graph cache tests .
Reset iteration count for performance test .
fix NPE
refactor to make code more readable
fixed small issue with if - condition
GT - 3001 ( ) code review - add missing file exclude .
Annotate / document tests according to comments from @ jglick
ui : don ' t run full decompilation on start
[ FLINK - 8699 ] [ checkpointing ] Create deep copy of state meta data to avoid concurrency problem with checkpoints
Product bundles model
sulong : LLVMToolchainNode should not return plain null but an LLVMPointer
Remove FastJar and use java . util . jar . JarFile instead
Watcher : Allow execute Watch API fields in request params
Add facility to index in random order
Make randomNonNegativeLong ( ) draw from a uniform distribution ( )
fix test error
Avoid I / O in DecodeJobTest
[ FLINK - 19815 ] CoFeedbackTransformation is not supported in BATCH mode
Reenable BwC Tests After ( )
port finding stack offset to method call
Fixed - RBuckets . get ( ) method doesn ' t work with custom codec .
Content - > string conversion fixed
Improved documentation of < code > mkdirs / < code > method
Data stream support for get field mappings API ( )
shutdown the disk check service when the server closed
Create extension dialog fix
Automatic code cleanup .
fixed compile error
Fix constant pool intrinsification to check for non - null .
Adding details in the throw exception on a missing field .
Compatibility with JSR - 354 final ( and its new Monetary singleton )
Fixed web service client . Added more package documentation .
Also dump what the client is seeing . See
Tests to verify exceptions and errors are properly propagated
Bump version + update release notes
Pass COMPUTERNAME env var to elasticsearch . bat ( )
Publish ApplicationContextInitializedEvent on contextPrepared
Enable - - incompatible_use_toolchain_providers_in_java_common
Fixed scene2d actions no being returned to the pool .
[ FLINK - 19299 ] [ tests ] Fix that NettyShuffleEnvironmentBuilder # setBufferSize does not take effect
Fix wrong exception in PolyglotProxy .
KristianRosenvold : Further detailed the asserts , maybe it ' ll make it clear what ' s happening . . .
[ ] Replace synchronization performed on util . concurrent instance in TrafficCounter
Change HttpServerUpgradeHandler . UpgradeCodec to allow aborting upgrade
adjust pg url for proxy
Add continuity counter support in Transport Streams
JsonWriter , fixed outputing quoted name / value containing a quote .
don ' t allow unresolved field types to hang around
[ TEST ] Made sure that the parent exists ( ` foo ` is the parent type and not ` parent ` )
for , refactor DropIndexOptimizedStatement
MINOR : fix Streams version - probing system test ( )
return false for HSAILRegisterConfig . areAllAllocatableRegisterCallerSaved ( )
[ VERSION ] Add [ 1 . 6 . 0 ] - SNAPSHOT
set default for Node . USE_GENERATED_NODES to true
Custom column filters fix
Make test pass
Update to latest jvmci version .
SQL Server : schema filtering fix
fix unstable UT ` AbstractMetadataReportFactoryTest ` under multiple network adapters ( )
Consider meta - annotations when creating an AnnotationsPropertySource
[ ] Port fixes from Akka to HashedWheelTimer
ArtifactManagerFactoryDescriptor
Replace health calls with authenticate calls ( )
Fix an ErrorProne warning about under - specified equals
[ ] Use IllegalArgumentException as replacement for NPE as stated in javadocs
Typo fix
[ FIXED JENKINS - 25897 ] Add range check for H ( X - Y ) syntax
Move method init into try / catch so exceptions are properly propagated .
don ' t do this by default
Wait for mapping in testReadRequestsReturnLatestMappingVersion ( )
Upgrade to RSocket 1 . 0 . 0
Fix nullability warnings
fix linting
A subtle bug repairing
rename to derivedCountIndexes
GraalCompilerTest : don ' t check exception messages that may have been omitted by C2
Allow MediaVariations . forMediaId ( String ) to accept null
Oracle : save object before compile
Solution for
SimonStewart : Fixing the typing test .
Provide page size from VirtualMemoryProvider to Unsafe ( instead of vice versa )
[ ML ] Fix ML memory tracker for old jobs ( )
[ TEST ] print hot threads when op job fails
Adjust init map size of user data of index commit ( )
Simplify annotation attribute override algorithm
Remove unnecessary okhttp close when canceling .
Do not force focus on spreadsheet after editor close
Minor reformatting to fit project style
modify pattern
+ UseOnlyWritableBootImageHeap now makes the heap section writable when used with + SpawnIsolates
fix TransformRobustnessIT intermittent test failures
Prefer profile photo over system contact photo .
fix compile error due to new signature of InternalTestCluster
Add ReplicaAction for Index Lifecycle ( )
minor fix
for , replace pg to 3rd generation parse engine
Fix NPE in PolyglotUncaughtExceptionHandler if Env is null
Table , set the cell widget size even when there is no widget .
Avoid exception when creating ChunkIndex with length = 0 .
Refactor StringDecoder
make removeMediaItem return void
[ ] Ensure using a ChannelInitializer via ServerBootstrap . handler ( . . . ) produce correct ordering .
[ GR - 20601 ] JLine uses a JNI library on Windows .
Polish " Align prefix match in BufferCounterService with DefaultCounterService "
[ FLINK - 5663 ] [ runtime ] Prevent leaking SafetyNetCloseableRegistry through InheritableThreadLocal
More tests for max bucket reducer
Fix regression in DefaultAttributeMap / Add Attribute . key ( ) / More fine - grained locking
add request overflow checking to OperatorSwitch
Data export from query fix
[ GR - 5832 ] Temporarily disable thread disposed assertion .
Use TestSelectWindow instead of TestClick as the sample test , because it does an array comparison
Add test for data split across multiple buffers
[ FLINK - 1437 ] [ Java API ] Fixes copy ( ) methods in PojoSerializer for null values
JSON formatter fix
[ FLINK - 10774 ] [ tests ] Test that Kafka partition discoverer is wokeup before closed when concurrently accessed
Adjusted timing of cumulative thred pool counter stream test
Fix test case issues
Fix deprecation warnings .
Add support for ` Execute ` keyword in Greenplum External Table DDL
DefaultHeaders / CharSequenceValueConverter should treat boolean consistently .
bug fixed for jsonpath ' . max ' & ' . min ' , for issue
HttpComponentsHttpInvokerRequestExecutor uses HttpComponents 4 . 2 to explicitly release connections
Support deploy_env in java_binary rule
Fix read date value
Fix ExtendedBeanInfo indexed write method edge case
Add char tests and fix String to char cast ( )
Truffle : some tweaks to the PrintTruffleExpansionHistogram .
bazel packages : simplify createRuleClassStarlarkThread
[ GR - 2091 ] Remove duplicate name getter functions in Sulongs LLVMFunctionStartNode .
Rename class for better clarity .
fixed typo
Fixed sporadic errors in SearchAfterIT : ack the create and index method calls in the search_after integ test
Fall back to the error drawable if model and fallback are null .
Fixes build errors around resource watching related tests
fix database show problem
Single node cluster comes up as red with local gateway , closes .
Query DSL : query_string / field / text do not fail when an invalid analyzer is provided , closes .
[ TEST ] Account for increase in ML C + + memory usage ( )
Core : Change < < < placeholder to { {
KAFKA - 5147 ; Add missing synchronization to TransactionManager
Fix failing tests
Atomic SQL operations fix
[ FLINK - 19533 ] [ checkpoint ] Add multipass StateAssignmentOperation .
Revert " Minor refactoring in variable naming in sctp stream config "
Eval method fixed .
[ TEST ] Improve SSLTrustRestrictionsTests ( elastic / x - pack - elasticsearch )
Replace MockAmazonS3 usage in S3BlobStoreRepositoryTests by a HTTP server ( )
Fix assertIngestDocument wrongfully passing ( )
SQLite : dattime data type support
[ GR - 21102 ] Throw a PolyglotException instead of an InternalError on invalid interop conversion to i32 .
Added uniqueness properties to inputs of KMeans example
[ GR - 5160 ] Make TruffleRunner more AOT friendly .
Fix multi - thread intents launching a conversation
added compilation time statistics to product mode
Fix a test case .
Fixed AArch64 relocation mistakes in SVM .
RSM filter update fix ( links navigation + pseudo attr visibility )
Make Android assets work in external repositories .
change version & fixed java doc .
Update MvcConfig . java
Also check for VirtualStackSlot when tracking busy spill slots
[ Discovery ] Handle ConnectionTransportException during a Master / Node fault detection ping
Polishing
[ GR - 7562 ] Improve synchronization not to create multiple binding nodes in multiple threads .
add the tcp socket close function
Support binary field type in script values ( )
Correct error in VertexBufferObjectWithVAO
Fix NPE
Disable unsupported features
java cuda compilation fix
HANA metadata queries
Implement pan / pitch / volume / loop
[ STORE ] Calculate Alder32 Checksums for legacy files in Store # checkIntegrity
SQL script commands model
Minor stylistic tweaks
[ FLINK - 12636 ] Check if proper JAR file in JobWithJars # checkJarFile
fix not thrown exception , pass inner exception
Search Hit Refactoring BWC
GraalCompiler . emitLowLevel : remove unused parameter .
Fixup after merge
Identifying association detect fix
Also dissallow null elements on set
MINOR : Wakeups propagated from commitOffsets in WorkerSinkTask should be caught
Translog : Reduce the number of translog ops to flush after from 20000 to 5000 , closes .
[ FLINK - 16813 ] [ jdbc ] JDBCInputFormat doesn ' t correctly map Short
AdaptiveRecvByteBufAllocator should ramp up while reading
Add support for # EXT - X - START tag
[ FLINK - 11436 ] [ avro ] Manually Java - deserialize AvroSerializer for backwards compatibility
KAFKA - 10186 ; Abort transaction with pending data with TransactionAbortedException ( )
backout 631329e3661a
Process final output buffer if it ' s non - empty .
Move shared attachment job generation code to parent class .
Add nullness annotations on the fields in ImmutableMultiset . java
Node Stats : Certain indices level stats to retain stats even when shard relocates , closes .
Fix license expiry logging
Fileset ( Google internal ) : code cleanup .
Oracle schema initialize fix
RootMethodSubstitutionTest should compare the high tier graphs
[ SPARC ] changed register counts due to hint
fix refactor getProductName
Lazily evaluate hash codes for ConfiguredTargetFunction # AttributeAndLabel .
Strengthen asserts
fixing typo in test method name
[ FLINK - 1686 ] [ streaming ] add parallel iteration test
simplified a little bit .
Improve error handling for epoch format parser with time zone ( )
Avoid creating a new lambda object for every FloatConvertNode
Remove " artifactOwnerConfiguration " now that LIPO is gone
Pass unused " millis " argument to lock . tryLock method , line 76 .
Skylint : improve error message for unused imports
[ streaming ] faulttolerance update
Error handle ( show connection error )
3 . x : Fix concatMapDelayError not continuing on fused inner source crash ( )
Don ' t include port in Host : header if it is a default port
Remove deprecated es . http . cname_in_publish_address setting ( )
for Refactor DataSourceParameterExtractor using AdvancedMapUpdater .
Fixes for new conditional elimination phase . Make it default .
[ fixed ] issue 514 , destroyBody was fluxored .
Address failing CCR retention lease test
joing method is already available in Util ( although I agree that it ' s not really discoverable )
Eliminate string manipulations when dispatching debugger events
Cleanup of CENC support
Expose return and arguments profile to subclasses .
OverflowStrategy in ConcurrentWebSocketSessionDecorator
Raised timeout for HystrixObservableCommand unit tests
Text and numeric formatting error handle
for , move isMaxRowCount to Pagination
Fix the logic of getting maxSuccessQps in StatisticNode ( )
JavaInterop : KEY_INFO message should not always be a TruffleBoundary .
[ hotfix ] Remove unused applicationName parameter from FlinkYarnSessionCli # createDescriptor
fixed spelling error
ConfigurationClassParser falls back to ASM - based resolution of member classes in case of NoClassDefFoundError
changed synchronization on Date type adapter to be for the format object per code review comments on r452
Disable polyglot access in launchers if - - polyglot was not provided .
Fix position reporting during ads when period has non - zero window offset .
[ FIXED HUDSON - 5072 ] use Util . rawEncode in getViewUrl so spaces are encoded
Some tweaks in NetJavaImpl
Fix check for management library
MINOR : Follow - up improvements on top of KAFKA - 5793
Use graph builder plugins in unit tests .
SPR - 5782 - BufferedImageHttpMessageConverter
Abstract UI moved to model
refactor OracleSelectParser
Fix for usages that are phi in rematerialization
Note some nonnull parameters
[ JENKINS - 42194 ] Code review reveals valid point , no schedules is not a warning
add a callback when shard level routing changes
gui : fix incorrect settings loading
Job canceling ( try to use interrupt )
Use client side component name when no server side mapping . ( )
Support handling frame queries for MetadataRetriever .
[ scene2d . ui ] Added wrapReverse setting to HorizontalGroup .
add debug message when adding and removing addressing from the transport client
Made JsonCodecLoader as singleton
Use correct color index for bubble chart
Java : Improving handling of marionette errors
Make unused Loads simplify in more cases
Use a well known type in ValueHostInteropTest
change function names .
skip all geo point queries in plain highlighter
Fix PolyglotEngine tests
Turn onIdleDelayed Runnable into compileQueueIdled method / event .
[ Dubbo - 4323 ] fix use AtomicInteger instead of volatile to inc ( )
[ FIXED JENKINS - 31915 ] Proxy settings in plugins page are ignored ( )
Change Mock proto compiler target type .
[ GR - 21077 ] LSP stability fixes .
Fix invalid cast in error handling of TruffleFile .
[ FLINK - 9243 ] [ tests ] fix flaky SuccessAfterNetworkBuffersFailureITCase
ERD : omit inherited attributes
SimonStewart and DanielWagnerHall : Make the selenium test runner pick a browser by default iff we ' re in dev mode .
[ MaterialButtonToggleGroup ] Clean up
@ Nullsafe : fresco / animation + imagepipeline / animated
Fix more JDK13 Posix Networking substitutions
Don ' t eagerly call TO_NATIVE on non - pointers returned from Truffle NFI .
[ GR - 16551 ] Allow port 0 , which chooses an available port , allow host - name only , more strict port check .
do not pass database type to EncryptSQLRewriteEngine
Disable inlining if engine mode is latency .
SQL editor fix ( errors after rename / move )
Report non - matching build - server stdout on native - image stderr
[ JENKINS - 42934 ] When you specify options you need to include CREATE or the file will not be created
SQL presentation panels visualization
DanielWagnerHall : Don ' t blur and re - focus already focused element in HtmlUnit fixing issue
Update the bullet tests
Resample float audio to 16 - bit by default to enable audio processing
[ GR - 26366 ] Incrementally canonicalize in the economy low tier .
ParameterNameDiscoverer may return individual null entries in an array
Fix CollapsingToolbarLayout . setCollapsedTitleGravity ( ) am : cd7313d4cd
ScrollPane , separate scrollTo and scrollToCenter . Fixed scrollTo x .
use type descriptor to determine collection element type
Revised test for ImportBeanDefinitionRegistrar double scan
Improve error message for installing plugin ( )
RSV metadata resolve fix
[ FLINK - 16219 ] [ runtime ] Made AsyncWaitOperator chainable to non - sources .
[ JENKINS - 22681 ] Additional logging useful for diagnosing certain lazy - loading bugs .
Deprecate SlowPathException # fillInStackTrace ( ) before making it final
First pass at adding tests .
for , remove useless codes for OrConditionFiller
Simplify DefaultChannelGroup . contains ( . . . ) and so remove one instanceof check .
Enhance FileOptions to support multiple classpath resources
Identifiers quoting fixed
using ListMap , org . greenrobot : eventbus - annotation - processor : 3 . 0 . 0 - rc
Read - only connections + modify permissions settings conflict fixed
re - apply linting
[ hotfix ] [ test ] Simplify RecordWriterTest
FS . get ( ) : Add URI causing exception to message .
Update ColumnarTranspositionCipher . java
Libgraal bridge .
Create trigger postgre fix1
Use correct stamp in BitLogicNode smart constructors .
version : change version to 0 . 5 . 2 - SNAPSHOT ( )
Nested attributes update fix
for , remove InsertColumnsSegment . type
Fixing HtmlUnitDriver to enter RETURN and ENTER to a text area . Fixes
KAFKA - 8586 : Fail source tasks when producers fail to send records ( )
Style fix
[ TEST ] Fix typo in MovAvgIT test
typo : prvoiderService - > providerService ; connectSkiped - > connectSkipped ( )
Fix data streams tests ( )
SimpleApplicationEventMulticaster just swallows event downcast exceptions
Fixed TABLE_ROWS column name in mysql partitions
Keybinding fix . Annotations
PlayerTrackEmsgHandler : Release sample queue in release ( )
DouniaBerrada : Fixing code formatting
fix a flaky test by using a FakeTicker
MySQL execution plan : query type detection fixed
prevent NPE
nested revision
Calling delegate on LoaderHidingResource # close ( )
Don ' t advertise support for video / mpeg ads
Adjust probabilities if some types can be ruled out via static analysis .
SWT error handle
Kirti Deo " Spring Webflux "
add json - processor support for non - map json types ( )
l10n fix
KAFKA - 5412 : Using connect - console - sink / source . properties raises an exception related to " file " property not found
[ GR - 4178 ] Check why Dump Whitelist does not work properly .
[ GR - 13701 ] [ GR - 14421 ] Avoid Concurrent Modification of Parser Scope .
Context constants
Fix invisible media controls and notch jank .
Remove adaptive thresholds .
Fix error tests , producer should wait a while for consumer to be ready
fix : use soft checks for objects and arrays in ' if ' type listener ( )
added ClassIsInterfaceNode macro node
better logging ( port number ) when failing to bind
[ GR - 10859 ] Page - align heap partition with relocations .
Provide methods to create implicit load / store converts without adding them to the graph .
SimonStewart : Add some missing tests to the selenium rc test suite .
Upgrade S - expr symbol alphabet .
TraceRA : verify that the simple TrivialTraceAllocator works
Exceptions from peer closing should not be propagated .
fix oss build + +
create main function GCD
Return actual status code not 200 to machine client
RObject . expireAt fixed .
Consider @ Deprecated on field when determining property ' s deprecation
Add ClientRequest . attribute ( String )
Provide File System documentation
Avoid unnecessary decoder instantiations on key rotation
GT - 3597 improved default reference update during language upgrade
Schema / catalog resolvers
Adding unit - tests to verify bugs .
Add test case for
Tweak offload disable Javadoc
Add a warning to Python host tools built for the wrong version
Stop needlessly parsing WORKSPACE files from external repositories .
Fix typo in warning
[ FLINK - 8177 ] Replace TestingContainer by mock in YARN UtilsTest
Support Opus and Flac in MP4 / DASH
Dialog , cancel method to not hide , restore previous focus actors , prevent focus leaving modal dialogs .
[ TEST ] Wait a bit longer before the lookback import has completed and it stopped the datafeed automatically
Reference bazel rather than blaze in user message .
Cache the result from ObjectMapper . getJsonValueMethod ( )
Add state in exception message ( )
Emphasize that java_xxx_proto_library . strict_deps should not be used for new code .
added grouping of permissions .
EranMes : Adding another variant of the script which can be returned from getAttribute ( ' onclick ' )
Date Field Mapper : Allow to use timestamp value ( milliseconds since epoch , UTC ) , closes .
Fix compile error introduced by 32231ee
Polish
GlyphLayout uses fontColor us default color in SelectBox
Copies from read - only heap ByteBuffer to direct ByteBuf can avoid stealth ByteBuf allocation and additional copies
Fix PixmapPacker - generated TextureAtlas disposal .
Fix starting the download service in the background throw exception
Fixed issue : NETTY - 359 Missing support for HTTP ' Expect : 100 - continue ' header .
bug fixed for db2 sql parser . for issue
Remove unnecessary code
AMD64HotSpotMove : split HotSpotLoadConstantOp into HotSpotLoadObjectConstantOp and HotSpotLoadMetaspaceConstantOp .
Optimize HttpContentEncoder to do less memory copies as those are not needed
object key & array maximum level limitation
Fixed an unforgivable mistake ^ _ ^
DragAndDrop , added x , y , pointer to Target # drop ( ) .
Improve error message for errors propagated in ExecutionEventNode . create ( ) .
Check for null when constant folding HubGetClass
Added note on non - thread - safety of Jackson ' s DateFormat support
Update MyActivity . java
Use full display name by default for model links to Items
Skylark : Add function ' int '
[ GR - 5393 ] Allow unstarted threads in the image heap .
Remove unnecessary null check
SwitchNode : avoid sharing any arrays while cloning
Support n meta - annotation levels on methods in AnnotationUtils
Fix for issue . Wrong dependency on log4j .
Fix PrimaryAllocationIT # testForceStaleReplicaToBePromotedToPrimary ( )
Make getter for bulk shard requests items visible ( )
Add exclusion for well - known property types
dbeaver - corp / dbeaver - i18n fix1
Set Format gapless info separately to Metadata in Extractors .
Rollforward of rollback commit 078c25c2f3cce5f407e28938ab2f3429e4609b19 with properly executable shell and correct test outputs .
SQL utils : full attribute name generation fix
Revert " [ TEST ] use logger level from test class annotation also in external nodes "
pull out string constants for settings , and add javadocs
LoadHub should be guarded by null check instead of range check
Firebird : table cache refresh fix
Implement SubstrateMemoryAccessProvider by HostedMemoryAccessProvider
[ FLINK - 19969 ] [ cli ] List ' run - application ' in help message
[ GR - 19493 ] More robust and informative graaljdk creation .
Correctly run pending tasks before flush and also remove incorrect assert .
Make ' deprecated ' a meta label ( not shown in the regular label list )
Reject invalid forwarded headers
[ tinker ] tidy up codes and modify resource patch logic to replace mResDir only for app ' s resources instance .
Add Jackson serialization inclusion to FactoryBean
Prevent ApplicationListener that depends on EMF from creating a cycle
Fixing UnitTest
only force fix an unguarded JavaReadNode
Added ellipse builder
Remove unused type parameter declaration in RequestResultMatchers
[ FLINK - 13440 ] Add test for FLINK - 12858
GT - 2898 : Code review fixes .
Fix handling of delayed AdsLoader . start
Logging : change logging to warning in LocalAllocateDangledIndices
[ FLINK - 19635 ] [ hbase ] Fix HBaseConnectorITCase . testTableSourceSinkWithDDL is unstable with a result mismatch
KAFKA - 9642 : Change " BigDecimal ( double ) " constructor to " BigDecimal . valueOf ( double ) " ( )
hideAll ( ) now resets all internal state .
[ bs - 136 ] Add ' spring run - - local . . . '
Fix parameter order for RequestMatcherAssert . doesNotMatch ( )
Android q ( )
[ streaming ] Changed GroupedWindowInvokable to always delete groups in case they have an empty element buffer
Update MySqlLexer . java
BAEL - 1107 : formating and version ( )
Fix InterfaceDesguar Comment Typos .
Design the runtime intrinsic factory to be suitable for inheriting from it
[ FLINK - 1409 ] Let CoRecordReader subscribe to buffer reader notifications
- Fixed unavailable current info when initial stream of playlist fails .
[ GR - 14745 ] Prevent from NPE on stop of the WebSocketServer .
Fix list partitioning helper .
SPR - 5507 Factored out the Phased interface .
Fix reference map for an IsolateThread to use uncompressed references
Fix a minor mistake in the generator and docs
Reduce Number of List Calls During Snapshot Create and Delete ( )
[ fix ] Callee saved registers should always be saved
Fix checkstyle issues
Remove a uneccessary var
[ GR - 16177 ] Fix spurious IllegalAccessException thrown by field getter / setter handle lookup .
[ JENKINS - 51355 ] - Print stacktraces to logs at FINE + level when rejecting classes in ClassFilterImpl ( )
minor formatting of LazyTailArrayNode
Override intConstantLength and branchTableLength for IfNode .
Correctly calculate and respect if we can correctly fullfil wrap for alerts . ( )
rm unused args
add getShardingMetaData ( )
Fall back to TYPE_ROTATION_VECTOR if TYPE_GAME_ROTATION_VECTOR unavailable
[ JENKINS - 47020 ] - Improve job search experience ( )
[ JENKINS - 14807 ] Fix path separator when EnvVars overrides variable like
Safer shutdown
Optimizing finding multiple elements by id by using CSS selectors if available . Fixes issue 7682
Fix a bug in ShowIncludesFilter . java
Follow 307 / 308 POST redirects for license requests
Query DSL : ` constant_score ` and ` filtered ` queries cache filters by default , remove it , closes .
[ JENKINS - 36908 ] Make sure that the All view is created ( )
TakeLastTimed with backpressure support
Relax assertSameDocIdsOnShards assertion
modify the cat
refactor OrchestrationShardingSphereDataSource
Complete Jetty Access Log configuration properties support
TESTS : Remove Static Threadpool in TaskManagerTest ( )
Prevent PhysicalMemory . size initialization inside VMOperation
Only set mapping resources when present
change version to 1 . 1 . 45
Automated rollback of commit c6b6dbadd0a93936c51154b25abc5fbba8f2d1af .
Optimize the hints at the beginning , when choosing a process
Use lock free strategy
[ hotfix ] [ tests ] Extend MockEnvironmentBuilder to support TaskManagerRuntimeInfo
Fixed incremental font when scaled .
Replace all database . beginTransaction with beginTransactionNonExclusive
[ FIXED JENKINS - 16630 ] Human readable file size method returns " , 00 " for
for , remove useless parameters of AbstractWhereExtractor . extractWhere
Always include ourselves in optimistic profile fetches .
add option for disabling the life time analysis
Rewrite EventTranslatorTest in spirit of JUnit 5
Add formatAsList method , and remove NodeSourcePosition requirement for inline logging .
Extract method for printing engine options
Complete backport of ( )
Simplify watcher indexing listener . ( )
Fixed potential concurrency problem
Remove media - session extension nullness blacklist
simplify YamlProxyConfiguration
[ hotfix ] Let DistributedCacheDfsTest extend TestLogger and avoid stdout output
bazel syntax : rename DebugServer to Debugger
ExperimentalEventHandler : narrow synchronization blocks
modify tags
NettyDataBufferFactory . join returns single - element buffer as - is
Allow moving media source within DynamicConcatenatingMediaSource .
Move assertion from canonicalization to constructor .
eliminate sign extension in AMD64StringUTF16Substitutions . indexofLatin1
Polish ApplicationContextAwareProcessor
Rename ' size ' JSON attribute ( of BitmapFont ) to ' scaledSize '
Bazel exports environment variable values to apple toolchains as build
Read - only inline editors .
[ hotfix ] Fix checkstyle violations in ExecutionGraphTestUtils
Fixes problem with ambiguity of Hadoop hostnames
Improve start_recovery check in IndexRecoveryIT ( )
Refactored StreamUtils a bit .
KAFKA - 4056 : Kafka logs values of sensitive configs like passwords
Removed unused var
Cleanup , dead code and formatting .
[ scene2d . ui ] Added font , background , and isOver protected methods to SelectBox .
[ JENKINS - 29798 ] Add a lock to serialize updates of tcpSlaveAgentListener
making the message more informative .
Switch back from method ref to Runnable in HlsSampleStreamWrapper
fixed @ Value injection to correctly cache temporary null results for non - singleton beans ( SPR - 7614 )
Avoid issue with switching from Reactor 2 . 0 . 5 to 2 . 0 . 6
Add more variants of ByteBufAllocator . ioBuffer ( ) / Update Javadoc
[ GR - 26173 ] [ GR - 26770 ] Fix blacklist violations related to ByteBuffer when building native images .
for , StarSelectItemSegment = > ShorthandSelectItemSegment
Fix backup timestamp language .
[ streaming ] Small fixes for StreamConfig
Omit useless argument
Fix propagation of HlsMetadataEntry ' s in HLS chunkless preparation
Oracle table column comment change fixed
Migrated remaining tests in lib / exec to JUnit 4 .
SimonStewart : ignoring new tests that don ' t pass for the selenium - backed webdriver
add a version to routing table , so it can be reused when not changed after publishing
[ ] Use SelectorProvider . open * ( ) to open NIO channels and so remove condition when create new NIO channels .
Fixed spotbugs warning .
[ GR - 3617 ] Some profiling remains in the compiled code .
, more compatibility
[ FLINK - 3714 ] Rename getCleanupTimeForWindow to cleanupTime in WindowOperator
Add support for Maven 3 . cmd files in Windows
For , Implement getProcedureColumns
Fix : Branch handling for incoming connections in UnionNode
Allow multi - value spring . profiles properties
Fixed event listener caching through equals / hashCode on SyntheticParameterizedType
Expose http address in cat / nodes and cat / nodeattrs APIs
addes SortMerger . close ( ) invocations to unit - test
Code cleanup
MINOR : Using primitive data types for loop index ( )
Minor bug fix in Truffle frame implementation .
AbstractBytecodeParser : move createTarget to GraphBuilderPhase .
Allow for reuse of key visitor classes in tooling
[ hotfix ] Some Java 7 cleanups in InstantiationUtil
[ FIXED JENKINS - 28690 ] Deadlock in hudson . model . Executor
Update ProjectFileSupport to take an OptionsParser .
Truffle : factor out assumption invalidate into a truffle boundary .
Update variable names that referenced old node1 / node2 names with new A / B names .
Node editor input persistence fix
add 7 . 1 . 1 and 6 . 8 . 1 versions ( )
refactoring
fix bugs
DerivativeReducer now copies histogram options from old histogram instance
for , fix javadoc
fix jad commnad VerifyError when the class is created by lambda expression .
LwjglPreferences , put back in . prefs dir .
fixed a test failure .
Remove assumption in test about / tmp location
Revert " Bug HexView resize by base 16 . "
Ensure native - image - maven - plugin executes java as java . exe
process reviewer comments
fix check on which settings to change on
PostgreSQL attributes length / precision . Attribute type editor .
[ FLINK - 13774 ] [ table - planner - blink ] Remove unresolved expression in RexNodeConverter
Fixed issue concerning linechart circle coloring .
Auto - configure templated welcome page
Add jar that should have been added as part of 69c61d0
Do not add index event listener if CCR disabled ( )
fix ConcurrentModificationException
More info about slave up / down in logs
Add before / after getText events ( )
C + + : Add more parameters to Skylark API needed for match_clif
[ FLINK - 19779 ] [ avro ] Remove the " record_ " field name prefix for Avro format deserialization
[ FLINK - 3292 ] Fix for Bug in flink - jdbc . Not all JDBC drivers supported
delete final String shardingSchemaName
Change Splitter # on ( String ) to Splitter # on ( char ) for single character strings
Snackbar , not animating only when talkback is turned on
optimized code .
Introduced isBroadcastChannel method in AbstractOutputChannel
update DownloadDialog . java
Polish
Correct doc reference tag ( )
add test
Fail at preparation when trying to download live content .
[ CLOUDBEES - 2018 ] improving the performance under a large side buffer .
Array values editor
MINOR : Fix BNF output for protocol arrays conataining primitives in docs
Minor code cleanup .
There might be other fields introduced by some tools , just ignore them
Remove InputStream constructor relate
Full Quartz 2 . 2 support , including LocalDataSourceJobStore
Turn logging of native - compiler usage into APIOption
calculate
Use null instead of ' none ' for ddlAuto
Change output path of the resource path shortening map .
Update ActiveMQConsumerInterceptor . java ( )
modify the processor
Refactor CppConfigurationLoader
style fixes
[ GR - 5872 ] Sulong interopt tests non - deterministically fail to lookup global symbols .
[ GR - 12309 ] Do not force class initialization when a method contains a lambda .
Remove unecessary calls to super ( )
Deferred handling of Flux error in Spring MVC
SELECT . . LIMIT fix
KAFKA - 4175 : Can ' t have StandbyTasks in KafkaStreams where NUM_STREAM_THREADS_CONFIG > 1
KAFKA - 8824 : bypass value serde on null ( )
aggregator and yaml tests for missing agg ( )
modify toSQL ( )
[ GR - 6198 ] Make closing of DebuggerTester deterministic .
Better verification of reachable Truffle types .
SQL : SubSelect unresolved bugfix ( )
split long line in ConvertProcessorTests
Reduce overhead of cancel flowcontrolled writes .
Try to stabilise descendantOfPropertyAccess performance test
Fix stats in slow logs to be a escaped JSON ( )
ERD table collector ( exlude views and other non - table stuff )
Increase startup timeout for Docker containers
use EncryptDataSourceBeanDefinitionParser
Extend HostActivity for reuse and silent timeout .
revise spring - boot jpa example .
changed README , assert has changed
[ GR - 14164 ] Make sure offset is sign - extended in HashTableSwitchOp .
remove redundant serialization test for JvmInfo and OsInfo and expand existing NodeInfoStreamingTests
Add already - implemented interfaces Closeable and Flushable to the cache .
Ensure that a non static top docs is created during the search phase
[ FLINK - 10377 ] Support checkpoint overtaking a savepoint in TwoPhaseCommitSink
fixed a test failure .
Throw exception on duplicate mappings metadata fields when merging templates ( )
Query manager view - copy fix
modify conditions for Preconditions .
Updated to use a Stream
AST can include inlined nodes .
Turn DuplexRequestBody into a marker interface .
refator top model
[ fixed ] seriously , i have no idea what i just fixed . See issue 224 on the tracker : D
Polish message method handling tests
Actually it was being taken care of . . reverted back
use aligned memory accesses for vectorized array copy
More AOT mode improvements
Fix generic warning in Plan
Follow up fix to close streams at the end .
change minute bug
Add new / missing version constants
Rename ClassloaderClassResolver to ClassLoaderClassResolver
Support JDK11 layout in LLVM toolchain launchers .
protect against an NPE in the call to setProperty to restore the old PI mode value
SimonStewart : Weeding out a couple of unneeded calls to setSelected and toggle
Start process of making auto_cpu_environment_group non - experimental .
Always return 401 for not valid tokens ( )
reimplemented the SCM poll locking algorithm .
for , refactor ColumnSegment
Refactor
GraalCompilerState : prepare Request on setup .
Rename " basic " configuration to " native "
Tolerate different connection failure exception thrown on Java 11 +
Do not zero out pages for bytes output stream ( )
fix signed vector extension to make intrinsic for vector comparison work
Use buffered tanslog type also when sync is set to 0
Support @ llvm . dbg . addr
Firebird procedure source generation
[ FLINK - 1652 ] fixes superstep increment in CollectionExecutor
Add SHA - 256 to SecureHashUtil
Split RequestMappingIntegrationTests into sub - classes
added Example test
Add handling for null values in PlatformProviderUtils .
MINOR : Fix potential integer overflow and String . format issue
[ GR - 26556 ] Host conversions should try lowest target type mappings before giving up .
[ FLINK - 7 ] Prevent range partitioning inside iterations .
Create target folder ( if not exist )
Always include extension renderers in SimplePlayer .
Fixed index out of bounds . Recalculate the line breaks when the size changes .
Fix the rest of the typo InMemoryRepository ' s name
Fixed JSON using Class . forName before tagToClass .
: persist icon size cookie across browser sessions .
Use correct native methods .
[ fixed ] Image TextureRegion was final . For whatever reason .
disconnect on exception when refreshing
for , use limitOffset & limitRowCount to instead of rangeItem_ for limit g4
reformat
[ JENKINS - 60351 ] - Make it possible to disable multiple deletion attempts by setting hudson . Util . maxFileDeletionRetries to zero ( )
polishing
Change minimum JDK version for compilation to 1 . 8
SimonStewart : Implementing attachFile for the webdriver - backed selenium
Add touch listener callback support .
Smart combo render fix
[ GR - 25918 ] Make tier name consistent in TraceCompilation output .
Split HttpDecompressionHandler into HttpContentDecoder and HttpContentDecompressor
Maintain order in headers .
use InsertValueToken
Rename " Save SQL Script " into " Export SQL Script " . The sme for Open .
Workaround to have multicast tests work on osx , linux and windows
[ ML ] The sort field on get records should default to the record_score ( )
core : fix indent for anonymous classes
Mark threads as read when we get an outgoing sync message
[ FLINK - 9567 ] [ yarn ] Before requesting new containers always check if it is required
[ FLINK - 17175 ] [ core ] StringUtils . arrayToString ( ) should consider Object [ ] lastly
Guard against out - of - range timestamp
, for comment : refine SQL execution event mechanism
Some details added to ConnectionPool error message
CommandsQueue object allocations optimization .
For tests that check against output paths , provide the ability to strip out the C + + - contributed fragment .
update
[ JENKINS - 9367 ] revisting the fix - - - changing the access modifier is an incompatible change .
LIRTest : support for zero parameters and subclasses of LIRTestSpecification .
Fix native - image image build for Java 11
Allow to access localAddress and remoteAddress even if the channel is not registered yet
[ FLINK - 1598 ] [ runtime ] Better error message when network serialization of records exceeds java heap space .
[ JENKINS - 22967 ] Fix item name semantics for DirectlyModifiableView web methods
Fix bug for dcl test
Cleanup Concurrent RepositoryData Loading ( )
[ hotfix ] Fix confusing error message in TableSourceValidation
remove null check for ProxyContext . getRuleRegistry
DouniaBerrada : Reverting parts of r14551 which removed RemoteWebDriver . log .
[ FLINK - 13946 ] Refactor executor code to ban resource sharing between plan executions .
Add equals and hashCode to MediaItem
JENKINS - 41778 - setup wizard issues when failures
| FIXED JENKINS - 9183 ] Maven3 with multiple threads does not work in Jenkins
formatting bug .
KAFKA - 3578 : Allow cross origin HTTP requests on all HTTP methods
cancel method .
debug
[ hotfix ] [ connectors ] Fix shadowed NPE in elasticsearch sink connector
Promote assert in getCodeAddressOffset to VMError . shouldNotReachHere
Fix host + port resolution in ReactorServerHttpRequest
Fix template _msearch with extra tokens
Abstraction for system - level virtual memory management
[ GR - 3746 ] AOT : Rewrite redundant initialization elimination .
Remove MapperService # dynamic . ( )
, DatabaseType . MySQL = > DatabaseType . PostgreSQL
[ GR - 20633 ] Normalized OS name string .
Fix [ FLINK - 2391 ] Storm - compatibility : method FlinkTopologyBuilder . createTopology ( ) throws java . lang . NullPointerException bug
Drop EMSG atoms before MOOV
[ hotfix ] [ runtime ] Unified variable names in StreamingJobGraphGenerator .
[ ML ] add default inference config to lang ident model ( )
Add / webjars / * * to default list of unsecured resources
[ JENKINS - 23365 ] Follow - up API fixes : introduce SCM . getKey ( ) , and add an SCMRevisionState baseline argument to checkout .
Log warnings when cluster state publication failed to some nodes ( )
Ensure assertion error is thrown for tests in case of null license ( )
Try to bring some more clarity into DynamicSpawnStrategyTest .
For , implement getCatalogs
fixed code generation error for debug builds
Limit CEA - 608 captions to 32 chars per line
[ scene2d . ui ] Respect background drawable min size when computing pref size .
[ GR - 13740 ] Enforce check for experimental options .
Center number display vertically for non - signal contacts .
Do not produce empty IDs in simple versioning test
fix run_under command
Fixed mysql auth
Avoid race condition in ILMHistorySotre ( )
KAFKA - 2674 : clarify onPartitionsRevoked behavior
Link remote folder - NPE fix
Implement ChipDrawable . isStateful ( ) .
Read only hex editor
Only reset the drawable in setScale if lottieDrawable is the current drawable
fixes When scanning super types , replace type variables with actual types .
Remove decoupling flag from Bazel
[ FLINK - 4754 ] [ checkpoints ] Small followups to the configuration of number of retained checkpoints .
Increase randomness in SocketUtils
AArch64 : Generate relocations to fit the ldr / str size
Deprecate BoolQueryBuilder ' s mustNot field ( )
Fixed indentation
[ GR - 5083 ] Do not add null elements to Source URI .
Add io . netty . noResourceLeak option to microbench
Added experiment to not release the state in unmount
Testing : Try to rebind in case of BindException
Use soft deletes for searchable snapshots tests ( )
SQL : More polish
MINOR : Always specify the keystore type in system tests
image - download - adding jsp and plain copy to response method
Polish
value phi node : fix error msg construction
add createEncryptColumnToken ( )
MINOR : Remove redundant braces from log message in FetchSessionHandler ( )
Spatial attribute transformers . Add extra settings
Results error status show fix
Test cleanup .
[ FLINK - 11524 ] [ tests ] Make JobResultUtils # assertIncomplete implementation more comprehensive
Fixed
Fix postroll content complete notifications
Add AndroidAccessToCollectors .
GraalArithmeticStubs defaults to false .
Update ChannelManager . java
[ TEST ] Disable wildcard query in testDfsSearchType for now
Test for placeholder - specified default value within @ Value
renamed test method and removed standard out output
improved mysql parser .
Wrap inputs to ObjcLink ( via clang ) invocations in an objlist , and pass the args as a - filelist arg .
Propagate EXT - X - DATERANGE tags with media playlists
Update version to 1 . 9 . 11
Implement whitelist outgoing edge backsliding
Move CCharPointerPointerHolder to nativeimage API
[ hotfix ] [ gelly ] Don ' t print Null edge values
Fix failing HeaderAssertionTests
Polish
core : fix variable definitions for ' try ' blocks
Disallow Skylark implicit outputs from referencing attributes
[ ML ] Improve handling of exception while starting DFA process ( )
TraceRA : hinting : propagate outgoing stack slots correctly .
Test the Gradle plugin against Gradle 4 . 10
Fix getDensity ( ) on iOS to be consistent with the rest of backends ( )
instead of isShowSQL ( )
Suppress source release failures .
Added indentation in the " main " function statement block
Parse source . properties using Properties class
[ GR - 26875 ] Weaken the assertion in Source # equals for now .
Upgrade to Hikaricp 3 . 2 . 0
Checkstyle fix .
CHANGE : Make HostAndPort GwtCompatible by getting rid of a regex .
, fix test case
Proper instanceof check added to PartialEvaluator . ParsingInlineInvokePlugin . hasMethodHandleArgument
Fix for SRC - 478
fix the cat client bug
remove clearParameter log error
[ hotfix ] [ table - common ] Relax literal casting check in AdaptedCallContext
[ hotfix ] [ runtime ] Clean up minor issues in TaskInformation
fix cygwin / mingw input support
JasonLeyba on behalf of TedMao : Fix parsing of user . js
RSV : data filter fix , add missing filter constraints
[ GR - 26511 ] [ GR - 26521 ] OSR improvements .
modify assertAddTableShardingValueForIn ( )
PG trigger create fix + new object create error reporting added
PgSQL parser ( code blocks recognition ) fixes .
Fix pre - sorting of shards in the can_match phase ( )
Make VerifyOopStub leaf call
ShardSearchFailure handling of exception does not take actual into account for status
Added getImageCell .
Copies shared object dependencies when building Android native binaries .
[ FLINK - 3438 ] ExternalProcessRunner fails to detect ClassNotFound exception because of locale settings
Updating for Firefox 1 . 5 . 0 . 2 fixing bugzilla bug 303713
Minor fixes , added PieChart transparent hole .
Properly handle virtual objects in the FrameState when checking for nested locks
SSHJ tunnel implementation
HttpServerRequest # setExpectMultipart ( true ) now throws an IllegalStateException when the request is not such a request - see
Fix incorrect name encoding / decoding in DNS records
Allow reporting unsupported MethodHandle invokes at run time
UriComponentsBuilder . fromHttpRequest sets port correctly
Documentation and more refactoring of Simple Language
remove unused code & add testcase .
[ FIXED JENKINS - 22693 ] - Properly close InputStreams in FileParameterValue
[ streaming ] StreamRecord getters refactor
Add NewInstanceServiceLoaderTest . assertLoadService ( )
Don ' t try to instantiate abstract classes .
Refactor integrate test codes 4th .
[ added ] SpriteBatch . not finished yet . Omg , Collections . sort ( ) uses iterators in Dalvik . gah . . .
Remember the name for LibFFISymbol to help debugging
[ FLINK - 20008 ] Close LeaderElectionDriver outside of lock in DefaultLeaderElectionService . stop ( )
use real setting instead of the settings object
Unused method argument .
[ GR - 10951 ] Print a warning if an instrumented node with null source section is found .
update version
[ TEST ] copied delete bw comp tests to usual intergration tests
SimonStewart : Migrate the android tests to junit4
[ TEST ] Upgrade test can only run if major or minor version differs
Do not mix int and long indices in MemSet node .
Make git revision loading lazy ( )
Pref pages < - > DS association fix
[ FLINK - 18310 ] [ metrics ] Properly handle interval parsing errors
[ GR - 21099 ] Fixed TruffleException . initCause .
Fix FlowableOnBackpressureBufferStrategy
Changed visibility of SIZE_LENGTH member
Changed Constructor of Ellipse ( Circle ) ( )
WebStatFilter support realIpHeader
Disabling the final OK check to move the integration forward
[ ML ] [ Data Frame ] write a warning audit on bulk index failures ( )
Use the correct service name in the request package
[ JENKINS - 18074 ] Updated comment .
ORacle : Numeric data type max precision fix
[ GR - 16728 ] Fixed deprecations on tools .
add another test to scan
Automated rollback of commit ad31fd31537f45ef4eacfe074e58bbcf20f1cfa9 .
[ FIXED JENKINS - 7725 ] mailto : links should be absolute URIs
[ GR - 3617 ] Some profiling remains in the compiled code .
Only check inode of image on isolate creation and improve errors .
Fix NioEventLoopTest # testChannelsRegistered flakiness ( )
[ FLINK - 18252 ] [ checkpointing ] Fix savepoint overtaking output data .
updates to make max http url length and the max header size configurable . Also exposed isSSL in http request interface
SimonStewart : FirefoxDriver will now throw an exception if unable to grab the lock
[ GR - 11064 ] Separate changes in indentation from printing of newlines .
Cleaned up release logic in RollingCommandLatencyStream
Fix buffer leak in test which was introduced while implement ZLIB_OR_NONE support . Related to [ ]
KAFKA - 4301 ; Add more trace for SSL handshake
Query DSL : Bool filter does not take should clauses properly into account , closes .
Prevent text tracks with no language being selected by default
[ fixed ] issue 570 , Polygon . getVertices ( ) used rotated x in calculation of rotated y .
More Java 8 in tests
Fix for encoding issue with MvcUriComponentsBuilder
Replace System Rules with System Lambda
Error handle fix
Simplify RoundedBitmapDisplayer
[ FLINK - 2845 ] Fix TimestampITCase . testWatermarkPropagation ( )
feat ( test ) : add support to run tests with different timeZones
Clear JNI pending exception for new native - to - Java calls and restore afterwards unless a newer exception was thrown
core : fix missing code after ' if ' inside loop
SimonStewart : Fixing an problem with smart DNS providers cause tests to fail
[ FLINK - 11775 ] [ table - runtime - blink ] Use MemorySegmentWritable to BinaryRowSerializer
KAFKA - 9523 : Migrate BranchedMultiLevelRepartitionConnectedTopologyTest into a unit test ( )
Catch ` Throwable ` in polyglot launcher
bazel syntax : avoid allocation in str . { starts , ends } with
MySQL driver upgrade + date / time default format fix
Tolerates thrift encoding bug we made a long time ago ( )
Fix typo in HistorySettinsFragment
Prevent a feedback loop between an animation and the seekbar in the sample app
fall back to app class loader for LayoutFactory provider on JDK 9
refactor for
Fix NPE in Partial Snapshot Without Global State ( )
[ FLINK - 8489 ] [ ES ] Prevent side - effects when modifying user - config
fix remove vertex
Expose that the LIR instruction list is an ArrayList
read elimination : respect unsafe access kinds during read / write caching , check for value stamps during merging
Upgrade to RSocket 1 . 1 - M2 and Reactor 2020 . 0 . 0 - RC1
BAEL - 4686 : Added some more test cases for TreeMap and LinkedHashMap
[ FLINK - 17040 ] [ tests ] Fix SavepointWriterITCase failure because of UnsupportedOperationException
test : fix test bug
[ hotfix ] Fix typo in AbstractMetricGroup . java
Switching to ACL . as .
JENKINS - 60513 : Added transient to SCMDescriptor # generation AtomicInteger to fix warnings ( )
Draw over controls fix
Use checkstyle check code
BuildToolLogs : clean up interface
Remove hasVtableEntry , use isInVirtualMethodTable instead
Remove - - proto_path = . from protoc command lines .
core : fix ' finally ' extract in ' if '
[ GR - 7143 ] Add slow path to the substitution of Array . newInstance .
Upgrade to Spring Batch 3 . 0 . 1
Return an empty string instead of " null " when the browserName and version
Fixed a bug that caused the miscalculation of yChartMin .
Use ConcurrentSkipListMap similar to StaplerDispatches
Fix QR Code Regression .
- Fixed timezone parsing when input starts with ' + ' sign . Fixes issue
3 . x : Remove a pointless concatMapIterable overload ( )
Fixed testIsValidKanjiSequence ( ) . Looks like the data arrays got copied over wrong from the C + + hex strings .
refactoring
[ Transform ] Fix casting in ExceptionRootCauseFinder ( )
[ GR - 23072 ] Host objects should not expose fields of non - exported classes .
fix test
fixed retriever cache putting
Fix typo in StringUtils import
Next development version
cleanup work
switch shard - level actions to static final vars , use separate method
[ tests fix ] Fix runtime exceptions for large record sorter by properly mocking the environment .
SQL Editor - handle connection delete
[ FLINK - 17665 ] [ network ] Replace isBuffer with BufferType field in BufferResponse
[ FLINK - 4309 ] Fix potential NPE in DelegatingConfiguration
Remove SYS CATALOGS leftover
Be more robust against load callback failures
Reduce parametrized runs for byte array benchmarks .
Allow MP3 files to play with size greater than 2GB .
return null if phase is not found in PhaseSuite . findPhase
Fix initializationData check for SSA subtitles
C + + : Migrate CppRunfilesProvider to new provider style
HashCodeTest . test08 : SubprocessUtil . getVMCommandLine ( ) may return null
Add multi uri support for Vito2
[ TimePicker ] Changed positive listener to be an OnClickListener
Fixed minor inconsistencies in log messages
Don ' t allow repeated calls to hash ( ) on a MessageDigest - based hasher .
Replace TruffleIntrinsics . deoptimize ( ) calls with CompilerDirectives . transferToInterpreter ( ) . Remove obsolete TruffleIntrinsics class .
[ JENKINS - 30084 ] more comments
Checkstyle fixes
Truffle / Instrumentation : comments and minor code cleanup post - review
[ GR - 19785 ] Addressing code review comments
Disable dummy tooltips
Add RxJava1 Reactive Streams adapters check
fix unit test case error .
polishing
Fix translog prepare commit and commit test
Add workaround for unmatched track indices in trex and tkhd .
KristianRosenvold : Fixed code formatting . In other news , IntelliJ Idea 110 . 137 now has working per - project code styles
Renamed Gson . getNextAdapter to getDelegateAdapter .
Handle the edge case of zero - length chunks .
Silence useless netty warnings in the remote worker
[ fixed ] hello world example . wtf .
Fix profileCount naming
added unwrapping of scoped proxy tp unwrapResourceIfNecessary ( ) ( SPR - 5671 )
Fixed test that relied on internal document ordering .
Simplification of boolean expression
Only support Safari on OS X .
Use nio driven http client for testing ( )
[ FLINK - 13047 ] [ table ] Fix the Optional . orElse ( ) usage issue in DatabaseCalciteSchema
trim threads in the right direction
Reset BytesMessage after payload extraction
Restore old CTW output format
Add MetaUtil . classForName
Fix incompatibility with XFS ( and possibly other file systems ) .
Tighten visibility
Temporary fix for Spring snapshot changes
ConstantStringIndexOfSnippets : Fix NPE in libgraal by copying byte [ ] with reflection
Ordering check fixed
[ hotfix ] [ test ] Rename and reformat testCancellationFailsWithBlockingLock
make NodeBitMap . grow public
[ texturepacker ] Allow empty pack . json .
Marvel : Fix MarvelSettingsTests
[ FIXED JENKINS - 12514 ]
[ fixed ] issue 630
[ GR - 15775 ] Minimize code size prior to verified entry point .
[ hotfix ] Java 8 - ify HadoopSecurityContext
Watcher : Support arrays in http response payload ( elastic / x - pack - elasticsearch )
Safer to use AtomicFileWriter .
Changed record serializer back to old version to maintain compatibility with old unit tests
Fix race in global checkpoint listeners test
BeanPropertyRowMapper uses US locale for lower - case conversion by default
Eliminate most uses of non - Unicode whitespace in com . google . common tests .
Hide bwc build output on success ( )
Identifiers quoting fixed ( escape only if name starts from _ )
Adds codecs for more list variants .
MINOR : move " Added / Removed sensor " log messages to TRACE ( )
Made HystrixCommandMetrics . getHealthCounts ( ) final
added both to XStream so that the on - the - disk format won ' t be messed up .
[ hotfix ] Add EXCLUSIONS set to ConfigOptionsDocGenerator
modify the login name
Version change
Relax assertion to allow MethodIdHolders that have pre - assigned ids
fix issue
Remove unused local which was always null .
Clean up merge commit changes
Clarify documentation for implicit deps and default attr values
simple printing in servlet ( )
River not recovered when using single node after shutdown , closes .
Async + ActionFileSystem : don ' t throw NPE
security : add a built in kibana_user role
SimonStewart : Removing some duplication from the htmlunit driver , and a copyright header that should never have been there
Remove mapper service access in field capabilities ( )
Avoid using StreamTaskListener . fromStdout .
Watcher : Fix put watch action ( )
for , inline ParseRuleRegistry . generalRuleDefinition
Add availabilityTimeOffset to last available segment calculation .
[ FLINK - 9575 ] Remove job - related BLOBS only if the job was removed sucessfully
Reuse existing OS checking method
Update jvmci import : Use ResolvedJavaType . isAllocationCloneable
Check for errors without object allocation
Remove hyperthreading multiplier to improve performance .
Implement ProcessProperties # getProcessID ( Process process ) for Windows
Simulate IO exceptions in all FlacExtractor tests
refactor ShardingDropIndexOptimizeEngine
fix : xor with boolean ( ) ( PR )
Ignore numeric shard count if waiting for ALL ( )
bazel syntax : tweaks to list and tuple
Make the debug overlay correctly handle scale types
Properly differentiate between UNBOX and AS_POINTER in NFI argument handling .
[ hotfix ] [ tests ] fix topic name in FlinkKafkaProducerITCase # testScaleUpAfterScalingDown ( )
Fixed - rename and renamenx methods doesn ' t change name of current object .
Minor cleanup .
3 . x : Fix Flowable . concatMap backpressure w / scalars ( )
Properties loading with ignoreResourceNotFound covers SocketException as well
Fix timing bug in DataBufferUtils : : readAsynchronousFileChannel
Fixed title bar touches getting the Window children .
Fixed invalid error message assert .
Work on realm - linedataset example
[ SPARC ] Replace call
[ GR - 3859 ] Expose flags ( nsw , nuwc , . . ) when creating nodes for arithmetic operations .
Signal NPE ObservableAmb FlowableAmb ( )
Revert Observable . Range to support 0 for count .
Wrap POJOs as TruffleObjects
Rename appearstoHaveNoObjectFiles to appearsToHaveObjectFiles .
Automatically decode DNS domain name to unicode
Always use padding in drawable state so that it can survive newDrawable ( )
instead of InsertOptimizeResultUnit , use InsertValuePlaceholder for rewriting
Checkstyle .
[ FLINK - 16225 ] Add JVM Metaspace Error assumption test
Driver manager dialog fix .
Restructured Nephele end - to - end test
[ SPARC ] Adding registers f0 . . . f31 to callee saved ; fixing allocator test , removing direct acquisition of Unsafe from SPARCHotspotBackend
Command context fixes ( last command cache )
Change ExternalFilesHelper # create to branch on whether it thinks it ' s being
When entry span layer is MQ :
Added Exception handing test to BatchEntryConsumer
[ FLINK - 4417 ] [ checkpoints ] Checkpoints are subsumed by CheckpointID not , by timestamp
Fix bogus message about sandboxfs staying mounted , even when it wasn ' t enabled in the first place .
BeansDtdResolver resolves spring - beans . dtd declarations to spring - beans - 2 . 0 . dtd file
[ scene2d . ui ] ScrollPane , fixed scrolling via mouse wheel when there are scrollbars in only one direction .
gui : fix ui tabs handling
Test : use URI so paths work on Windows
Stop reading encoded cache key from EncodedImage
Reset min / max frames unless no composition is set ( )
Update thread timestamp on draft creation .
GraphML support ( draw . io )
[ FLINK - 13369 ] Track references of already visited object in ClosureCleaner
Mention ` - - incompatible_disallow_empty_glob ` in the error message .
Propagate Android aspect across runtime_deps .
Add register persistent worker latency metrics ( )
draw line by path to prevent thick line looks fractured
add description to retry compilation DebugContext
Fixed - RedissonSetMultimap . removeAll removes reference to nested set .
making sequenceEqual private methods public
test cleanup
Add an option to disable - - { host_ , } javabase = < absolute path > .
WallFilter refactor
Correctly determine the wat2wasm binary on Windows .
Completed propagation of checkpoint state changes
Automated g4 rollback of changelist 336326618
fix exo2 buffering state
make search fagment handle video items
SmartLifecycle retrieval fix , properly taking FactoryBeans into account ( SPR - 6545 )
, update name for SQLExecute span
JAVA API : Fix source excludes setting if no includes were provided
store for backup
Mappings : Fix doc values representation to always serliaze if explicitly set
[ GR - 13953 ] Support providing a custom ImageHeapProvider and CommittedMemoryProvider .
Search page update
Add support for ObjectSizeEstimator : in order to allow special optimization if possible from user code
Using a MessageDigest per Hasher , not per MessageDigestHashFunction ( making the latter thread - safe )
[ FLINK - 16139 ] [ runtime ] Reset colocation constraints when restarting tasks in DefaultScheduler
Rename writeThrowable to writeException
fix bug Date format in StateAnalyzerTest
bug fix .
make MathGame faster
Fail fast if @ WebAppConfiguration and @ SpringBootTest are used together
Removed index level metadata election
Use of API method to create an empty list .
Ignore URI when local . mongo . port is set
[ GR - 21602 ] Cleaning whole Lazy instance .
modify the cat
Address PushObserver feedback .
Implemented basic version of open ( path , bufferSize )
+ fixed NPE when closing up non - initialized contexts
RSV status bar style fix
SimonStewart : Starting firefox on windows should be better now ( issue )
PlayerTrackEmsgHandler : Release sample queue in release ( )
Add key mapping for numpad numbers
fix : exclude directories from dex convertion results
fix resource not close problem . close
Remove logic of entering context when entering resource in annotation pointcut
[ GR - 6958 ] Test the publicaly visible aspects of the JMX bean .
[ hotfix ] [ checkpoints ] Improve performance of ByteStreamStateHandle
BAEL - 89 remove csrf disable
FileStreamChannel should be compatible with all event loops
[ GR - 15418 ] Suspension filter is applied to breakpoints as well .
Fix for suggestions in PullRequest :
Expose ` search . highlight . term_vector_multi_value ` as a node level setting ( )
Log [ initial_master_nodes ] on formation failure ( )
[ FLINK - 15995 ] [ table - planner - blink ] Change TO_BASE64 operand type checker from ANY to STRING
Allow setting transparency amout for transparent circle / hole ( ) .
For checkstyle
don ' t dump the graph in SchedulePhase during lowering
Return null info if service is not started
Polish
Fix wrong type in LLVMAddressMessageResolution .
Added comments to level order traversal method , and samples in main method .
RSV filters fix
Start generating HTML from Span - styling in SubtitleWebView
Rename LoggingInitializer to be consistent with others
[ inliner ] readability
Cardinality request breaker leak ( )
delete isOperateIndexWithoutTable ( )
Cleaned up and adjusted BulkIterationWithAllReducerITCase
fixed compile error
Use ThreadLocal . withInitial to init ThreadLocal
Finished implementation of channel throughput measurements
Use WeakReference instead of PhantomReference .
Defensively handle non - retrievable ApplicationListener bean names
Provide error message if - - debug - attach is used more than once
Fix bug in ActionSchedule .
[ changed ] SpritePerformanceTest2 a little
[ yarn ] [ python ] Adds - Xms start option for yarn , fixes python test case to overwrite files , fixes python docs
Use Pollexor 2 . 0
Change toMultiset ( ) to accept . . . Function < ? super T , . . . > .
[ JENKINS - 31162 ] @ oleg - nenashev ' s comments were addressed
Add nullness annotations on the fields in LinkedListMultimap . java
[ a11y ] Ensure CollapsingToolbarLayout surfaces its custom - rendered title to accessibility services .
fixes and debugging
Exempt " Test " action mnemonics from needing to have Xcode related execution requirements set .
[ FLINK - 12316 ] [ jdbc ] Relax assertions
fix : error loading resource map file from bundled jar ( )
Ensure SSLErrorTest also works with boringssl
Blacklist Moto C from setOutputSurface
Update signal - service to 2 . 5 . 8
Extract HlsMediaChunk # feedDataToExtractor method
improved the default behavior of the newInstance method .
Configure NettyStreamFactoryFactory by default if available
KAFKA - 9297 : CreateTopics API does not work with older version of the request / response ( )
Incrementally canonicalize in the economy low tier .
polishing
GT - 3437 - Version Tracking - fixed exception when changing options
Fix bug displaying an empty camera contacts search result .
added more methods to IOSMonotouchGLES20 implementation
MINOR : Add support for Standalone Connect configs in Rest Server extensions ( )
Add realistic hlrc response serialization test base class ( )
1 . 0 . 23 - SNAPSHOT
Remove dummy parameter ` includeAbstract ` from ` ResolvedJavaType # resolveMethod ( ) ` ;
put error details into exception instead of printing to the console
message
configure live tests
Sibling of enforce access to translog via engine
Internal : promptly cleanup updateTask timeout handler
Distinguish between threads paused during initialization , and threads paused in response to a explicit PauseThreadsRequest .
Column type modifiers fix ( picture , timestamp )
AndreasTolfTolfsen : Ignoring AvailableLogsTest on Opera and Opera Mobile
Update PolygonFactory . java ( )
[ TEST ] Remove content type auto - detection while parsing request body in REST tests
for use multithreading to access database for Statement
lucene 4 : Fixed TermFactory usage in MapperService
If a build has no change , e - mail shouldn ' t link to the empty changeset page .
SQL completion enhancement ( column names prefix for * )
Refactored TakeUntil operator
compiler bug in IntelliJ ?
Fix broken build
Use decoders with names that don ' t start " OMX . " .
Pasted source code conversion
see 05 / 07 log
Enhancements to background playback and media button handling
Move depset order validation from PyCommon to PyStructUtils
Weaken assertion in ZenDiscovery . publish
Adjusted LocalInstanceManager to work with multiple temp directories .
Fix truffle LibraryFactory # clearNonTruffleClasses for Java 11
SerialSubscription and Timeout operator
Simplified code
fix cat start bug
[ FLINK - 10406 ] Add meaningful message if no savepoint location for cancel with savepoint
Made threadPoolForLoad non - static and transient , so it is recreated for every new Hudson .
[ JENKINS - 55257 ] Verify fix in test .
BuildView # getConfiguredTargetForTesting respects null transitions .
[ GR - 26081 ] Reset DynamicObject layout cache after native image generation .
[ FLINK - 18042 ] [ tests ] Auto - detect distDir
Fix error in Gradle build due to TextAppearance variable scope
Form validation for the remote FS root of slaves was not functioning .
Using a DP size for the debug indicator
Allow disabling SO_LINGER on java . net . Socket
more feedback
KAFKA - 5720 ; Fix AdminClientIntegrationTest # testCallInFlightTimeouts
MINOR : WorkerUtils # abort : fix bug in abort logic ( )
Added TiledMapTileMapObject .
Using global time stamp as a fallback execution id
[ TextInputLayout ] Adding setBoxBackgroundColorStateList method and fixing setBoxBackgroundColor method not changing focused and hovered states color like it did originally .
TNS names parse fix ( remove comments )
Motivation : ( )
Add support for more augmented - assignment operators .
[ FLINK - 3014 ] Replace InStr . read ( ) calls with DataInStr . readFully ( )
Ignore secure decoders . They shouldn ' t be explicitly listed .
Elements may not have been specified
remove synchronized in NewInstanceServiceLoader
Use XING headers without size / table of contents .
Remove vestigial serialization bits from ErrorInfo
Filtering schemas fix ( generic driver )
Query DSL : ids filter / parse to automatically use types provided as part of the search , closes .
[ GR - 6969 ] Temporarily disable AVX512 usages .
[ wasm ] Provide entry - point detection to the Wasm launcher .
Blacklist OMX . SEC . mp3 . dec and OMX . brcm . audio . mp3 . decoder on Samsung devices
Giving proper indentation
[ FLINK - 19949 ] [ csv ] Unescape CSV format field delimiter character
Make sure ExecutionEvent is always delivered
Inline XContentBuilder # writeValue
Test status quo for @ Inherited annotations in AnnotationMetadata
Speed Optimizations :
Catch option error during execution too , since OptionSet is passed there
MySQL table props
Adjust autoscaling metadata serialiation version
some post changes on the can ' t play fix .
Add more logging for testSnapshotAndRestore backward compatibility test
Replaced obsolete cryptographic primitive with a modern / secure one . ( )
Add data stream support to the reindex api . ( )
Polish CookieWebSessionIdResolver
Fix extracting Class constant when the target class is substituted .
[ FLINK - 1363 ] [ runtime tests ] Fix race condition in ExecutionVertexCancelTest . testSendCancelAndReceiveFail
BAEL - 3943 Bson to json ( )
Set EXTRA_EMAIL in order to make email clients fill in email address
add getKeyGeneratorProperties ( )
SimonStewart : Fix an obvious problem with the get attribute method ' s fallback
fix right ( ) bug
minor bug fix
Solves issue ( )
Add comments and tests
Fix getPolyglotGuestBindings might end up on the fast - path so it needs to deoptimize if not initialized .
Thrift : Rename Status . CONTINUE to Status . CONT ( reserved word ) , closes .
bugfix : fix auto proxy for multiple - datasource does not work ( )
[ GR - 2407 ] CompilationWatchDog : Disable per default .
Fixing console annotation masking in LogTaskListener
refactor : remove server - addr validator
Fix NPE in ErrorsArgumentResolver
Watcher : Replace System . currentTimeMillis ( ) with nanotime ( ) ( elastic / x - pack - elasticsearch )
FlowControlHandlerTest invalid condition
Make EOF_STATEMENT_SEPARATOR XML friendly
Setting HtmlUnit to emulate FF24 by default
Use built - in constants for HTTP status codes .
Move TARGET_CPU from the CcToolchainProvider to BuildConfiguration .
LogFormatUtils . formatValue leniently handles toString ( ) exceptions
Fix dashIgnoringElementEquals algorithm
refactor : removes the request forwarding logic for the configuration module
Remove unnecessary use of Reporter .
Property editor : specific props support
Truffle : further fixes to the new CallNode .
Inline ENABLE_PRELOADING
Connection settings hint
Propagate ASwB aspect across robolectric implicit deps .
Replace unnecessary StringBuilder calls with String concat
Revert " Fixes : Restore window on correct monitor " ( )
change connectErrorRetry default value .
Fixed bug in uniqueness determination in optimizer node
KAFKA - 8298 : Fix possible concurrent modification exception ( )
Update doc for the ` HTTP_METHOD_SPECIFY ` property item ( )
TraceRA : fix string concat issue in calls to Debug . logAndIndent .
Adding node_count to ML Usage ( ) ( )
Move createActionCache to ExecutorBuilder .
Use ValueConsumer in HotSpotBackend .
[ GR - 12953 ] Use equals ( ) to compare ResolvedJavaType .
Schedule load for non - primary playlist for low latency streams
Issue 1788 additional fix for vCard delimited name parsing
[ inlining - 2 ] make returned value explicit
[ GR - 10139 ] Installer uses catalog URL without version .
Fix issue with sticker preview sizing .
Updates SnapshotDeletionsInProgress version number introduced to 5 . 2 . 0
Added a POST with a request body and a DELETE method for HTTP with cURL demo
Only propagate tabIconTint and tabIconTintMode to default icon views , not to custom icon views .
BAEL - 3271 : Use random port in spring - session - redis IntegrationTest ( )
Change SpawnAction members back to private
Annotate safeContains and safeRemove parameter as @ Nullable .
TNS names parser fixed
use TLSCert values instead of paths for keystore
Check if controller is used when performing click directly .
Make array comparison nodes extensible
fix memory leak in HystrixPlugins ( )
[ ] Zero out private key copied to ByteBuf before release .
Test : apply ` shield . user ` property only once ,
[ FLINK - 3651 ] Fix faulty RollingSink Restore
Automatically autowire a bean with one constructor
Array type resolve . Nullable arrays .
, refine ErrorResponsePacket
SantiagoSuarezO : fixing , luckily there ' s no more need for this killableprocess thing
bug fixed for float / double to int / long .
sorting algorithms benchmarks
refactor ChannelThreadHolder from static to singleton
testRetainAll tuned
[ GR - 2927 ] Allow primitive arguments for PolyglotEngine . Value . as method .
Fix the serialization of sparse immutable tables .
Refactor SagaSQLExecuteCallback .
Disable disappearing messages option and remove from menu .
Fail if an error is not thrown .
Cleanup index settings updaters
Automatic code cleanup .
Return empty scopes even if env is null
SpringValidatorAdapter properly handles HV - 5 - style list constraint violations
SQLite : temp tables & sequences .
snippet template : make lookup go over loop begin nodes
Don ' t trace the collector endpoint
new import
count all type check hint misses not just the last one
Better message text for ResponseException
OpenSsl . memoryAddress ( . . . ) should use internalNioBuffer ( . . . ) if it can ' t access the memoryAddress ( )
Polish contribution
Reuse opened editor for data search
Relax test termination for API level 23 and above
StringUtils . parseLocaleString accepts Java 7 variants
[ JENKINS - 62231 ] Optimize catch code
Fix checkstyle
Don ' t switch from Output tab
GT - 2881 : cppexporter now respects cancel commands
Move BAD_ACTIONS precomputed value into SkyframeActionExecutor .
SL / Instrumentation : Added checks for null SourceSections in SLInstrumenter
[ FLINK - 16245 ] [ tests ] Adjust BatchFineGrainedRecoveryITCase
Few iterator improvements
Alphabetize PoolConfig . Builder
Automated rollback of commit d7e497a175e17498344f90c06ebe529e7bac1937 .
refactor getProxyDisabledDataSourceNames
[ GR - 11772 ] Correctly initialize 64 - bit versions of the 32 - bit values used to zap heap chunks .
simplify JDBCResourceManager
removed unused code
Don ' t create groups for 1 : 1 MMS conversations
Keep backwards compatible with 0 . 90 . 0 on the transport layer .
PostgreSQL : schema DDL fix
* Fixed a bug in HttpTunnelingClientSocketChannel where the HTTP request is not sent if the scheme is HTTPS
resolve NPE
Declare static types as Queue instead of Deque .
Fixes
- fixed double close of SpillingResettableIterator in CrossTask
[ TEST ] Add additional logging to testCorruptTranslogFiles
Document the terminatedBy methods
Default floating numbers rounding fix
Updated liquibase dependency to 3 . 2 .
refactor Bootstrap
Don ' t do a deep comparison in ArtifactNestedSetKey .
SQL Server : indexes and keys batch reading
[ TEST ] Also wait for fields to have been applied in the mapping in cluster state during teh waitForConcreteMappingsOnAll call
Native numbers formatting fix ( revert to toString )
Make InliningPhase reentrant .
add unit test case for merger 9th version
fix bug
fix : use CountableListener to replace RPromise
Make breakpad file editing platform independent .
Fix crash bug with the ' visible ' query function and non - package_group targets .
PEA : allowed for defered effects on ends in MergeProcessor
merge defaults from setDefaultOptions ( ) into GraalOptions
implement ASTExtractHandler1
[ ] Allow clients to override userDefinedWritabilityIndex from AbstractTrafficShapingHandler
Increase number of recent conversations shown when sharing .
[ fixed ] loading jawt . dll on Windows before anything else in JoglNativesLoader now . Crossing fingers : p ( tested on Windows 7 )
bugfix : restore autocommit ( )
Truffle : print inlined call target and rename profiling option .
Fix testUpdateRolloverLifecycleDateStepRetriesWhenRolloverInfoIsMissing ( )
Replace JobControllerImpl with interface JobController
[ docs ] Clean up completeness checker for configuration docs .
[ TEST ] use logger level from test class annotation also in external nodes
Support MPEG - TS streams that start / end with an incomplete TS packet or lost sync .
[ JENKINS - 38187 ] Fix latest comments
Optimize getAllItems ( )
Updates to splitting summary .
[ GR - 10832 ] Mis - translation of ` errno = 0 ` .
add ComputeLoopFrequenciesPhase after canonicalizer
[ java ] Fixing NPE in reactor HttpClient if the server returns a response with no content
Adding new tests ( two of which are disabled , but prepping for 1 . 7 ) .
Allow HTTP decoding post CONNECT in ` HttpClientCode `
Add missing logging param
[ FLINK - 17727 ] [ e2e ] Re - enable " [ FLINK - 17467 ] [ task ] [ e2e ] Modify existing upgrade test to verify aligned savepoints " after a fix
[ hotfix ] [ table ] Rename to TypeLiteralExpession . getOutputDataType
Fix SmoothStreaming Timeline
use source field declaration order in generated Node classes
reduce footprint of nodeRewritingAssumption
KAFKA - 3002 ; Allow uppercase letters in hostnames
[ Security ] Handle non - existent user in native realm ( elastic / x - pack - elasticsearch )
[ GR - 4069 ] Fix java stack frame method names must not be null .
[ ML ] Use the correct timeout for the process context lock ( elastic / x - pack - elasticsearch )
Polish " Add session property for ConfigureRedisAction "
Comment local attribute of genrule and remove TODO
Workaround LG - related menu issues , pt . 2
Reordered the address book fields to something a little more standard / reasonable .
Increase timeout for testAddNewReplicas
Skylark : Late - bound attributes may return None .
[ ] Correctly handle trailing dot in DNS requests and responses for the hostname .
Fix for empty datasource - name in metrics
Protect loop from infinite loop in the highest of zooms
Swig generated files
[ FLINK - 1201 ] [ gelly ] SSSP with runVertexCentricIteration
Optimizing VolatileReadNode out causes wrong memory state to propagate to uses of the VolatileReadNode
add router xml api
BAEL - 1171 String . lang . String API ( )
add example of request - response model
Add workaround to let the sleep work correctly in windows too . See
SPR - 7604 - Support multiple embedded databases .
[ GR - 23932 ] AArch64 : Add Math . multiplyHigh intrinsic on AArch64 .
Don ' t update apply glViewport every frame .
DB2 : Typo
Fixed issue 948 , also fixed LongMap .
Fix
Enable local action execution statistics collection for sandboxed actions that use the DarwinSandboxedSpawnRunner .
FlacExtractor : add condition for zero - length reads
PostgreSQL triggers
core : omit redundant brackets in case blocks
Split up rendering in the bullet tests
Remove the last usage of the " AAPT " enum value
JasonLeyba : CapabilitiesComparator should give preference to configurations matching the current platform .
[ java ] More conversion of guava - style code to java 8
TEST : write ops should execute under shard permit ( )
@ AutoCodec JavaCcLinkParamsProvider .
Do not provide substitution - deleted methods via reflection .
[ FLINK - 10777 ] [ tests ] Update TypeSerializerSnapshotMigrationITCase to cover restoring from 1 . 7 . x
added another callback
[ TEST ] fix failing DocumentParserTests tests
Move force ( ) after multiple writes , not at every steps
Issue 537 , don ' t return UPC - A for EAN - 13 starting with 0 when UPC - A isn ' t allowed
[ hotfix ] Extract loadJarFile from PackageProgram # < init >
AndreasTolfTolfsen : Made a mistake thinking UNIX should be considered Linux , updating copyright notice
[ FLINK - 8704 ] [ tests ] Port ScheduleOrUpdateConsumersTest
Handle overflow and negative timeouts :
this method can be static .
LinearScan : make opIdToIndex private .
Cache flush ( . . ) tasks which are used when executing with different EventExecutor to lower GC pressure
[ BAEL - 15393 ] - Initial commit to change to boot 2
Next development version , 1 . 9 . 7
Deleting declaration of exceptions that can never be thrown ( they are wrapped to WebDriverException )
Return 0 area rather than NaN .
Skylark : int ( ) can now declare a base parameter .
[ FLINK - 5701 ] [ kafka ] FlinkKafkaProducer should check asyncException on checkpoints
Use thread local random for request id generation ( )
Prevent from calling onLoad ( ) multiple times per a single Source .
Improve caches endpoint format
ScrollPane , clean up .
[ scene2d . ui ] Added Progressbar updateVisualValue and round methods .
better handling of failed merge because of different types
remove shardingRule and parameters for select parser
Fix native - image buffer overflow on windows using Chinese Charset
use Collection < Connection >
Add local . management . port when on same port
replaced = = with . equals ( )
[ FLINK - 17789 ] [ core ] DelegatingConfiguration should remove prefix instead of add prefix in toMap
fix problem detected by - G : - PartialEscapeAnalysis
Add additional compilation - final state to TargetOffset .
QualifierAnnotationAutowireCandidateResolver matches regular bean name for FactoryBean dependency type as well
make GreedyInliningPolicy / ExactInlineInfo public , allow optional null check suppression on inline
Respect < description > within < entry > per beans XSD
[ FLINK - 8747 ] [ bugfix ] The tag of waiting for floating buffers in RemoteInputChannel should be updated properly
[ GR - 18860 ] Recompile conservatiavely when branch target estimations are wrong
Fix exit code in certutil packaging test ( )
modify the heartbeat analyzer
reenable - XX : + GraalCountersExcludeCompiler option
set missing create param in PutRequest
Create mark file for local repositories , too
Support time_zone on composite ' s date_histogram ( )
use columnClauseStartIndex
Reduce object creation for for unwrap / wrap if no ByteBuffer [ ] is used .
Fix proxy language .
[ hotfix ] [ runtime ] Simplified and tweaked TaskMailboxImpl .
Fix template query parser tests to register mustache script engine
[ JENKINS - 48770 ] new test and using TimeUnits
Update async dispatch check in OncePerRequestFilter
Added SqlSessionManagerTest . Fixed getMapper ( ) implementation to not use the proxied mapper .
Fix a minor bug where RRBuildFilesVisitor wasn ' t using an
[ FLINK - 13982 ] [ runtime ] Generate dynamic configurations and JVM parameters with TaskExecutorResourceUtils .
KAFKA - 4722 : Add application . id to StreamThread name
Use a fixed thread pool in ByteStreamBuildEventArtifactUploader
do not add 1 to hour when generating display string for time and datetime column
Refactor hybrid helpers to more precise Hosted types
Use Imgur ' s size options in the comparison app
Attempt to fix " No active shards " failure
Fix compilation from FsInfo # < init > change in core
Fix issue with some search results returning empty .
Java 2387 ( )
Enable nullness checking for DefaultMediaClock
Stream TargetPattern # eval implementations ' results to a callback rather than returning a ResolvedTargets set .
[ TEST ] Reenable RareClusterStateIt # testDeleteCreateInOneBulk
[ GR - 22457 ] Refactor monitor slots .
Fix Target Api errors :
clean log
Minor changes , see
Reduced code duplication of AbstractXnioChannelHandler
Fix the ' compose ' unit test
Removed left actor from Tree .
Unignoring file uploading test in marionette
Remove unused copyWith methods
bug fixed for Timestamp serialize
Fix compressed OOPs .
Remove AnimatedDrawableFactory from PipelineDraweeController
Attach Instruction Metadata to the Intended Targets
Rename ` ChannelHandlerContext # fireUserEventTriggered ( ) ` argument from ` event ` to ` evt ` so it matches the ` ChannelInboundHandler # userEventTriggered ( ) ` argument ' s name .
Provide flag to disable throwing of cache initialization exceptions
Tests : Tweaking static bwc tests to improve stability
Bump the supported class file version for JDK13
support nested service types
SimonStewart : Error messages need spacestoo .
Prevent to set URL . factory to null in case of concurrent URL creation in the URLBuilder .
added limit for parsed SQL cache to NamedParameterJdbcTemplate ( SPR - 7237 ) ; added configurable cache limit to CachingMetadataReaderFactory
added memory usage to CompilationStatistics
Eliminate safepoints also in non - counted loops if they contain invocations .
Rename Torture test to Stress test .
Added ` clipValuesToContent ` property for clipping values
rename directories and META - INF / services files along with core library desugaring
MOVED , ASK handling in cluster mode using RBatch .
JavaDoc improvements
Fix forward propagation of guarded nodes in ConvertDeoptimizeToGuardPhase .
Add PostCodeInstallationTask to the backend , and have Truffle use it .
Fixes Keras import issue ( 8373 ) ( )
Upgrade to Hikaricp 2 . 7 . 8
Allow to set null handlers on HttpClientRequest and HttpClientResponse on end
* Improved the detection of chunked encoding
Avoid retrying when client is missing body File
[ GR - 11775 ] Fix NPE when dumping with 4 + level .
[ TEST ] don ' t use multiple names ( e . g . aliases ) pointing to the same concrete index when using indexRandom
[ GR - 2544 ] Unsafe access of edges should use wrappers in Edges class .
[ ] Fix NPE in AbstractTrafficShapingHandler which could happen if the READ_SUSPEND attr was not set yet
HOTFIX : fix off - by - one stream offset commit
Lock reentrancy race fixed .
Fielddata : IntArrayRef is initialized with small array and grows if needed
EranMes : Some CookieImpl tests failed because cookies were set on a domain .
fix the metric bug
Register a bean for OAuth2AuthorizedClientService
rename properties in RangeShardingAlgorithm to partition . ranges ( )
Added a way to change the Assets loader base URL for some special cases , overriding the getPreloaderBaseURL ( ) method .
Reduce scope of AST lock for visitRoot .
Set the LibraryIdentifier before building the LTO Indexing link action
fix auto completer problem
Bad connection issue if autoCommit = false
Reverted c93230f31b0d5e700e018261a1418780b5efa60c
RelativePathTest fixes
Add SinglePeriodTimeline constructor to fill missing window information
Help javac with generics
Add byte offset variable to fake data segments .
[ SPARC ] Make CAS delayable
option for dumping only partial evaluation results
Switch x - pack / plugin to new style Requests ( )
[ FLINK - 12051 ] [ runtime ] [ tests ] Wait for TaskExecutor to be started
Allow return without expression , AST - equivalent to return None .
Let Skylark rules take part in MakeVariableExpandingRule expansions using the
[ FLINK - 2536 ] [ streaming ] Cleanups and improvements on SocketClientSink
[ JENKINS - 27607 ] Noting merge of .
DB2 system schemas
RSV drop - down filter UI fix
Use the default values from SharedTruffleRuntimeOptions for integer values
Revert KeycloakConfigurationTest ( )
Add a " score_mode " parameter to rescoring query .
Synthesized events don ' t demand the cursor to be inside the browser window
Sulong : Make AsForeignNode uncacheable
Return the correct Future from FixedChannelPool . release ( )
queryParam options for MockMVC requeusts
[ GR - 2536 ] Reduce dynamic usage of ResolvedJavaField . getName .
[ GR - 6654 ] Improve UTF - 8 conversion in NFI on SVM .
[ Test ] Remove obsolete job / cluster cleanup code
Allocate buffer from ByteBufAllocator in ChunkedFile
MethodInvokeTypeProvider lazily invokes target method ( avoiding deserialization exploits )
Fixed the cascade of the " 2nd instance " problem .
KristianRosenvold : Removed timeout in grid while the request is being forwarded to the node .
optimize : downgrade check unlock and asynchronous ( )
RedissonTopicTest . testReattachInClusterMaster test added
removed JumpNode and the non - materializing instanceof snippets that used it
bug fixed for toString
Take into account 0x00 as a type identifier in peekBlockType .
properly handle failures by closing file refs
PG : drop index fix
[ FLINK - 9504 ] [ logging ] Change the log - level of checkpoint duration to debug
Tests : Enable check index in tests again .
SimonStewart : Closing issue by applying the patch from James Strachen that exposes the webdriver instance
Stronger check in translog prepare and commit test
[ GR - 17303 ] Inline linkToSpecial if the target can be intrinsified .
[ streaming ] Made constructor of GroupedDataStream public
Polishing
[ JENKINS - 26391 ] Prevent lazy loading operation when obtaining label information .
PROPFIND redirects should maintain request body
fix send icon merge botch , encryption icons logic
fix test case
modify the status
Add support for new codecs parameter string
Change some variable names
Fix buffering state when selecting VTT track with no cues
Fix for no paradigm .
[ GR - 21198 ] Fix Windows launchers in JVM mode .
Add an - - allow_empty_bootclasspath argument
Adjust serialization version for can contain data
[ GR - 8408 ] A couple of fixes enabling Sulong / FastR .
Fix test
Enable inline tracing in the bytecode parser .
Propagate end of stream received from ` OnFrameRenderedListener `
ResolvedJavaType . isAssignableFrom must not be called with null argument
Temporary fix for broken client code : allow NULL for Isolate and IsolateThread
Catch ActivityNotFoundException on contact add intent
Tested unsubscription , too
set setColumnClauseStartIndex ( )
Put NodeEntry in Supplier < SkyValue > passed to EvaluationProgressReceiver so it ' s readily accessible in the program .
Removed bogus optimizations added in 0b313004bd12d3aed7a0b744684fbccd57344115 .
[ FLINK - 4870 ] Fix path handling in ContinuousFileMonitoringFunction
Reset playback info but not position / state in release
Plaintext : pref page settings re - layout
dbeaver / dbeaver Dynamic proxy detection model
2 . x : Workaround for Objects . requireNonNull inserted by javac ( )
Added GeoDistance test which verifies the difference in behaviour between ARC and PLANE , causing elliptical results
SQL indent fix
Avoid re - using completed previous requests if skipMemoryCache is true .
Don ' t rely on ConfigurationPropertyName exceptions
fix for bug 56 where Gson wasn ' t serialization null fields of type string ,
DruidDataSource add keepAlive support .
[ java ] Adding assertj magic instead of type casting
KAFKA - 7902 : Replace original loginContext if SASL / OAUTHBEARER refresh login fails ( )
CTW shouldn ' t override Config options
Fix crash when dumping null values in the LLVM backend
Use colorAccent for Catalog Favorite Toggle
fix for OSR_MIGRATION_END descriptor
Fix a mappings update test ( )
Minor refactoring to SkyframeExecutor progress receiver .
wait for the start of a build more properly
Watcher : Do not encode URLs in HttpClient
[ FLINK - 10774 ] [ tests ] Refactor Kafka tests to have consistent life cycle verifications
Make sure the log config works .
Implement LLVM abs intrinsics for pointers .
Revert test logic back to original
resultsets use the trace level , not debug
[ FLINK - 4162 ] Fix Event Queue Serialization in Abstract ( Keyed ) CEPPatternOperator
Smart combo fix ( workaround for NPE in SWT )
LowLevelLowTier : add options for optimizations .
[ FIXED JENKINS - 28192 ] Uses java . nio . file . Path # toRealPath instead of File # getCanonicalPath , which resolves symbolic links also on Windows .
Set the MessageConsumer handler to null during unregistration
piggy back JFR inlining events on TraceInlining events
Add ` http ` and ` transport ` simplified host settings , closes .
HTTP / 2 Log Cleanup
fix test
Generate SQL for tables and RS
Stop importing ImmutableFoo . Builder .
update spring orche mybatis
[ FLINK - 1201 ] [ gelly ] typeinfo in reverse and getOutdegrees
[ GR - 19087 ] Inlining : Properly re - route anchor usages of exception object nodes when inlining into a graph with unwind .
BAEL - 4415 correct line continuations indent to 2 spaces
delete initServerConfiguration ( )
SQL ordering fix ( do not use indexed columns for type attributes )
Makes sure Elasticsearch urls are allowed to be plain text on port 80 ( )
Refactor ChunkedWriteHandler to remove synchronization which can have bad side effects like deadlocks . See and
merging 3 of them to make them more concise
Do not pollute Cache with failed futures
MECARD TEL should have only digits
Add inheritance test that references parameters .
[ GR - 19633 ] Fix NSP and probabilities information in ` If ` nodes generated during ` InstanceOf ` nodes generation .
Truffle - DSL : refactor ElementUtils # findAnnotationMirror
fix exoplayer thread
modify AUTHENTICATION_YAML content
Fix issue that can hide slow network requests
Fixing Test Cases
: Dubbo ' s ExtensionLoader may be can ' t find custom Extension ( )
Fix incorrect usages of getSourceRoot and getExecPath
Update MultiProducerSequencer . java
Fix Incorrect Concurrent SnapshotException on Master Failover ( )
Use the DeoptimizingNode interface of the Invokes in the LIRGenerator
Migrate to ApplicationContextRunner
DanielWagnerHall : Fall - back to expected default locations of Chrome in case the registry key is missing
Split HlsSampleStreamWrapper . init into two methods
[ BAEL - 3646 ] Addressing broken PR
Fixed - CommandDecoder doesn ' t remove command with failed response from commands queue .
do not add table that not in sharding rule 3rd version
Output TraceInlining format , and separate case when there are multiple decisions about callsites .
Round up parsers should be based on a list of parsers ( )
Fixing broken CrossDomainRpcLoader
SPR - 8688 Don ' t use Servlet request params for form ' PUT ' in ServletServerHttpRequest .
[ hotfix ] [ k8s ] Wrap podsWatch as Optional in KubernetesResourceManagerDriver .
fix log message . exception first , parameters later
Migrate ExoPlayerTest to use TestExoPlayer .
Partial rollback of changelist 151480429 .
Same fix for Netty V4 for issue
Explicitly pass hosted options in reportUserError
Painless : Fix regex lexer and error messages ( )
modify the config
Free up all buffers after the tests
Add test case ensuring that a BytecodeParserError is raised when an assertion is used within an intrinsic .
Polish
Enable disappearing message controls
modify the stateanalyzer
less aggressive tail duplication
FrancoisReynaud : keyword return missing . firefox binary passed as param was ignored .
for , remove EncryptOrPredicateFiller . isEncryptExpressionType
Make Groovy support feature more selective
KAFKA - 8058 : Fix ConnectClusterStateImpl . connectors ( ) method ( )
[ grid ] Fix incompatible type and make the build work again
Extended tests for delimited input format .
Fix a potential memory leak in schedulePeriodically
[ FLINK - 10364 ] [ tests ] Fix instability in NonHAQueryableStateFsBackendITCase # testMapState
Add more timers , fix style .
[ FLINK - 12193 ] [ tests ] Add option to disable failOnCheckpointingErrors
Reuse connections of the same address in ZooKeeper data - source ( )
[ GR - 26125 ] Make InlineDuringParsing effective in unit test .
[ CCR ] Make index . xpack . ccr . following_index an internal setting ( )
Revert " Merge branch ' dev - all - changes ' into dev "
use getTableRuleByLogicTableName ( each )
Fix StompSubframeDecoder . readHeaders produce not any notification when parsed line that contains multiple colon
JENKINS - 33805 - rename install / upgrade files to be more clear &
Fixed padding of the integer and node classes .
SQLite foreign key resolve fix
Compilation fixed
Add remoteApplicationName in RpcContext ( )
Improve SqlProvider test to be a more complete example
Geo : Issue with polygons near date line
Support override enable auto - configuration
Prefix Glide thread names with " glide - " and
[ fixed ] updated check in FastImage . Bollocks .
Fix checkstyle issue .
Cleaned up some logs
Rollback of https://github.com/google/ExoPlayer/commit/e26a61b903739014271ab0886bb5db75afb3f1fd

*** Original commit ***

Improve `Format` propagation within the `MediaCodecRenderer`.

For example, fix handling of pixel aspect ratio changes in
playlists where video resolution does not change.

Issue:#6646

***

PiperOrigin-RevId: 282903626
modify the jsp
Bump version to 2.11.0

Note: Release notes are not final.
PiperOrigin-RevId: 279125474
modify the unused transction
Bump version to 2.10.7

PiperOrigin-RevId: 278658259
fixed issue #136 , support rds no primary key && mysql.ha_health_check
for #601, removed useless codes for StatementAdapterTest
:recycle: Refactoring code.
bug fixed
add BenchmarkMain
fix #81
add factory
modify the decorator
Translate some exception message.
clean code
add the readme
fix regex error
fixed #191 Using IP lowest bit to generate worker id
Rollback of commit 9de9f374cf63e8e6938c9ce80820c9f46f641241.

*** Reason for rollback ***

Broke ci.bazel.io

See http://ci.bazel.io/job/bazel-tests/BAZEL_VERSION=HEAD-jdk7,PLATFORM_NAME=linux-x86_64/379/console for error log.
See https://cr.bazel.build/7610/ for tests that confirm fix.

*** Original change description ***

Pass isExternal field to blaze ide info proto

--
Change-Id: Ieff9fbc719ea7baf8321bf50570fae5f7d6c7fdd
Reviewed-on: https://cr.bazel.build/7610
PiperOrigin-RevId: 141029777
MOS_MIGRATED_REVID=141029777
Rollback of commit 30469a9939c14f8d0d2708bd0a91525a2e161cc4.

*** Reason for rollback ***

Breaks density filtering.

*** Original change description ***

Collect all the temporary directories into one tree that will delete on exit.

--
MOS_MIGRATED_REVID=114122848
remove currentTable
Rollback of commit d03485f1361ff825a976512959438321442c7c98.

*** Reason for rollback ***

Broke the build: http://ci.bazel.io/job/Bazel/262/

*** Original change description ***

Skylark: implemented more string functions (isalnum, isdigit, etc).

--
MOS_MIGRATED_REVID=110383960
Import publicsuffix.org's test data.
Run the tests as part of the tests of InternetDomainName.topPrivateDomain(), suppressing those that fail.
(Plus miscellaneous no-op changes to the test.)
-------------
Created by MOE: http://code.google.com/p/moe-java
MOE_MIGRATED_REVID=58354304
refactor sql parse engine
Update CurrencyTest5.java
fix issue
see 02/16 log
Fix test case bug
fix #88
replace fastJson with jackson for naming instance
#1609
update SimpleExoPlayer.Builder
modify the alert title
fixed testcase.
update MySQLServerInfo (#8111)

* update MySQLServerInfo

* update MySQLServerInfo test

* update download

* Revert "update download"

This reverts commit 91f208e8

* update download
[Issue] #1769 Solve the bug of the clone configuration function
Bump version to 2.8.4

-------------
Created by MOE: https://github.com/google/moe
MOE_MIGRATED_REVID=209134753
Rollback of https://github.com/google/ExoPlayer/commit/75eab31d79aae73196a60fb24474da2a36855c4a

*** Original commit ***

Rollback of https://github.com/google/ExoPlayer/commit/0943886cbd7ee6c4e33ed731ddff0b23e5699abd

*** Original commit ***

Use last queue format instead of previous decision to select new track

We currently use the save...

***

PiperOrigin-RevId: 320015109
refactor PreparedBatchStatement
Rollback of commit 8bae4f4e99594118b7ce651f7779f92e7683cdd0.

*** Reason for rollback ***

Reverting this because we need to revert commit 4b73e972d909bcd533f2f9940f95a00b9b73bdde. (See unknown commit).


*** Original change description ***

CppCompileAction must return false from inputsKnown for .d pruning

This is necessary (but not sufficient) for the action cache to work
correctly. Consider the following sequence of events:
1. action is executed
2. .d pruning is performed
3. action cache writes entry with post-pruning inputs list
4. action gets regenerated (e.g., due to server restart)
5. the action cache calls inputsKnown(), which returns true
6. action cache checks entry from step 3 against pre-pruning inputs list,
   which re...

***

--
PiperOrigin-RevId: 147831409
MOS_MIGRATED_REVID=147831409
Rollback of commit 6d98f6c633ec4a12737544a149bc18c41a3c442a.

*** Reason for rollback ***

Revert source file symlink resolution in XcodeGen. Since source files are not inputs for the XcodeGen action, this does not work if the action is sandboxed.

*** Original change description ***

RELNOTES: Resolve symlinks when calculating the workspace root in XcodeGen.

--
MOS_MIGRATED_REVID=94223656
fixed heartbeat
fix Issue #40 : parser context contains Null object. ex: $.null.beans1[0]
fix comment error.
related to 3288
[FLINK-13547][table-planner-blink] Refactor CONCAT() and CONCAT_WS() to keep it compatible with old planner

CONCAT(string1, string2, ...) should returns NULL if any argument is NULL.
CONCAT_WS(sep, string1, string2,...) should returns NULL if sep is NULL and automatically skips NULL arguments.
for #2082, recover cannot run test cases for old parse engine
Rollback of changelist 64659484 ("Make common.reflect build and pass tests against both Java 7 and Java 8").

Our internal build changes seem to be causing problems.
-------------
Created by MOE: http://code.google.com/p/moe-java
MOE_MIGRATED_REVID=64665073
see 05/14 log
see 10/30 log
modify the version and fix bug
[MI] Use mysql master heartbeat to detect phycial tcp connection
failure.
v1.5.7
* readme
bug fixed

git-svn-id: http://code.alibabatech.com/svn/fastjson/trunk/fastjson@1372 fb902949-7fc4-4485-a80f-4d6ff335c879
Bump version to 1.0.11.
sm support completion. #323
update
Bump version to 2.11.4

PiperOrigin-RevId: 303776645
Misc changes (#2637)

* refactor app namespace duplicate message

* fix non-properties namespace display issue
improved mysql parser. for issue #2028
bug fixed for issue #1884
see 11/27 log
fixed issue 1903 and added associated testcases
see 09/21 log
fix typo and arrange package
see 01/17 log
see 05/14 log
fixed #313
improve sql parser