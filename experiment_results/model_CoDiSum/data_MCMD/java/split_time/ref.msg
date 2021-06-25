hotfix docs conf add query service port to port section
bugfix range protection
updated icense years ci skip
fix initial brightness value
moved tests as well
fix floattorawintbits not using numberutils for gwt
get brightness from settings if screenbrightness is set to auto
using non abbreviated local variable name
update current frame bci when instrument is not null fixes stepouttest stepoutstayonline
make isolated namespace creation reachable only on linux
workaround for impossible cast warning
add at since todo to javadoc co authored by oleg nenashev o v nenashev gmail com
update umsresourcecategoryservice java
avoid checkstyle stackvaluegetclass warning
use linkedhashmap for a deterministic order
hotfix runtime small docs comments cleanups in rpc code
hotfix core fix broken javadoc link in fileutils
update umsresourceservice java
output is created at init annotate it as lazyily initialized
sulong add tostring to llvmparserresult
save task action label fix
sql value completion fix check column prefix
parsablebitarray add readstring
notify instrument on guest exceptions
give info if content language is system on crash
migrate annotation to androidx on recaptchaactivity
fix ehcache format
escape quotes in quoted arguments
polish javadoc for clienthttpresponse
merge branch dev into dev
invalidate before updating usesmonitors compilation final
report wrapped stackoverflow and oom exceptions also to instrumentation
put error handling behind transfertointerpreter
update pmsproductcategoryservice java
apply suggestions from code review co authored by jesse glick jglick cloudbees com
sulong javadoc to the per module initialization nodes
use stream api to collect deprecated plugins
fix instances of double checked locking
fix gradle plugin validation error
apply check for bci larger than the first bci on the last line before create call frame
read all bytes at once when parsing sourcedebugextension attribute
apply suggestions from code review co authored by adrien lecharpentier adrien lecharpentier gmail com
create a frame descriptor to hold a monitor slot for native methods
don t re assign the input variable
update pmsbrandservice java
mute indexlevelreplicationtests testseqnocollision signed off by mark vieira portugee gmail com
hotfix datastream api minor javadoc deprecation cleanup in checkpointedfunction
transformer dialog fix hide if no options found
mute recoveryduringreplicationtests testresyncafterprimarypromotion signed off by mark vieira portugee gmail com
mute debmetadatatests test05checklintian signed off by mark vieira portugee gmail com
update taskdispatchperiod default value
update cmsprefrenceareaservice java
fix if parentheses
fix generated sql using new cast needed for pgsql reg columns
update pmsproductattributeservice java
update mailsendtask java
update pmsproductattributecategoryservice java
remove void singleton
add sanity asserts
mute aggregatortestcase testsupportedfieldtypes signed off by mark vieira portugee gmail com
improve watcher test by preventing a npe when closing the http client
checkstyle error fix
checkstyle error fix
update umsmenunode java
remove unnecessary round floor int division seems better anyway
fix intset contains
remove unnecessary semicolon in some enum classes
fix merge issues
fix missing transfertointerpreter in generated code
updated license years
adjust test for new permission changes manage and system read permissions
update src main java org apache ibatis executor reuseexecutor java co authored by iwao ave harawata gmail com
downloadservice onstartcommand nullable intent
downloadhelper createmediasource nullable drmsessionmanager
hotfix mesos test update test cases for validating mesos deployment respecting the cpu configuration fallback order
hotfix minor clean up in taskexecutorprocessutils
hotfix fix typo in flinkkafkaconsumerbase
fix propagate language in debugvalue execute
the generated provider class of language instrument registrations must be final
minor javadoc fix
migrate wasm to new meta data api
fix invalid visiblity for inherited object
fix missing truffle boundary in sl
fix wrong import
fix javadoc formatting
fix typo in trufflelanguage javadoc
fix wrong usage of views in receiver source sections
fix missing boundary
fix ismetainstance needs a mention of asvalue coercions for the argument
fix missing truffle boundaries in tests
improve javadoc of trufflelanguage getlanguageview
make jcachemanager constructor public signed off by michael cheremuhin michael cheremuhin worldapp com
fix rebase issues
codestyle signed off by michael cheremuhin michael cheremuhin worldapp com
avoid need for boundary when saving input values
update umspermissionnode java
copy as typo fix
minor documentation fixes
exception is ignored in searchfragment
add isinstanced method to indicate whether this mesh uses instancing
unbind method must call disablevertexattribute
fix incorrect indentation in textfieldmapper
newpiperecyclerview should allow scrolling down by default
fix containskey for a couple maps
increase inlining and exploration budget
merge branch master of https github com alibaba fastjson
tables backup wizards fix save settings on page deactivate
rename table_max_pages to table_max_size
rethrow error when error message is not expected
add returns length check
add start function check
add invalid memory index in data segment check
fix public static order
changed drop down selection widget to avoid npe in multi selection widget
change might to can
remove unused import
fix non ascii char
fix unused import
sql completion value proposals fix
fix delegation to primitive types is now supported
validate activepeer during bluetooth and speaker audio state transitions
remute timeserieslifecycleactionsit testhistoryiswrittenwithfailure signed off by mark vieira portugee gmail com
fix a typo
fix log error
add missing binary source comment
use numfunctions instead of numfunctions
remove comment line break
javadoc update around sl invocations
use correct option to disable compilation
randomdir name fix
fix generic target type mapping was using a cached case with wrong assumptions
解决当data是linkedlist时 随着数据量增加性能变差 入参为list 当list是linkedlist时 get方法性能很差
remove deprecated isobjectoflanguage method
update to the latest getrawframe api changes
connection configuration storage model
remove unused imports
fix unit test case formatting
change frame access to read_write
hotfix fix checkstyle violations in scheduledunit
disabled survivor spaces by default
improve webclient builder javadoc
corrected testassemble_stmcsda_r2_lr0r1r2r4r6r7r8lc failure
add framework for two tests in scalar model
don t backup the keyvaluedatabase
hotfix tests make schedulertestingutils constructor private
use linkedhashmap for deterministic order
typed workbench access
fix for a ripple effect on a button
update cmsprefrenceareaproductrelationdao java
set mute button in main player from other player
update doc comment
hotfix hdfs fix error message when using hadooprecoverablewriter on non hdfs
fix backup restore crash
don t skip verifycompiler unless h exitafterrelocatableimagewrite is set and llvm backend is used
make referencequeue substitution public and remove enqueue method
mute nodesubclasstests testreplacechildren signed off by mark vieira portugee gmail com
mute rundataframeanalyticsit teststopoutlierdetectionwithenoughdocumentstoscroll signed off by mark vieira portugee gmail com
mute localindicescleanertests testhandleswatcherhistory signed off by mark vieira portugee gmail com
tests fix incorrect notify method
tests updated test to handle html escaped
update copyright years
sulong better exception message
import ioexception instead of using java io ioexception
make getargumentvectorblocksize usable for argument vector support checking
fix synchapi sleep signature
update cmssubjectproductrelationdao java
sulong fix memberexists to return false rather than throw an illegalstateexception
hotfix dataset javadoc fix typo typo in inputformat
close stream after loading resource
update app src main java org schabi newpipe settings videoaudiosettingsfragment java co authored by stypox stypox pm me
add notice on how to run the live tests
working with scope has to be done behind a truffleboundary
tinker fallback to use custom classloader again to avoid more compatibility issues
tinker fix build failure
remove unused option
add a new stronger test than the jck test for pop frames
sulong fix invalid format specifier
sulong make interop pointer constructor public
sulong rename getparameterlength to getnumberofparameters
sulong remove assertion that is superseded by guard
sulong llvmmanagedpointer must not point to null
re add old wasmbenchmarkstate api for compatilibilty with graal entreprise benchmarks
rename wasmcompilationbenchmarksuite to wasmmemorycompilationbenchmarksuite
update omsorderoperatehistorydao java
add doc comment
wrap datainputstream readfully call in try finally block
config profile delete usage checking fix
remove summary tag from javadoc
add a truffleboundary on locationimpl valueequals
add license header
add boundaries around unsafe park unpark
fix checkstyle compiler warnings
make interop constructor return receiver
add simple interop tests
stop start reference drain thread only if multithreaded option is enabled
fix klass getmembers
fix improve uncached dispatch
document single threaded mode limitations
expose members only for guest class instances
fix rebase debris
fix test fix the issue that messageclientidsetter getipfromid return wrong pid when pid greater than short type max value
generalize arraycopy loop expansion
add memoryprofiler doc
sql server constants
checkstyle errors fix
sulong the nfi should not need to handle libpolyglot mock
update pmsmemberpricedao java
remove unused imports
remove debug output line
hotfix tests fix parameter error in checkpointcoordinatortestingutils
remove redundant transfer to interpreter for slow path exception
null risk issue
annotations with identical encodedvalues but different visibilities are no longer viewed as the same annotation during patch generation
update encoder java
mute internalenginetests testnoninternalversioningonprimary
sulong make isinternallibrary more robust
mute regressionit testtwojobswithsamerandomizeseedusesametrainingset
adapt xpath to handle additional classes in html
mute gatewaymetastatepersistedstatetests testdataonlynodepersistence
suppress unused warning
fix semaphore used during local cache clearing process isn t deleted
show admin monitor alert when system_read
update pmsproductattributeservice java
tregex fix formatting issue
tregex fix matcherbuildertest
tregex fix erroneous word boundary snippets in pythonflavorprocessor
revert enabling benchmark
mute classificationit testtwojobswithsamerandomizeseedusesametrainingset signed off by mark vieira portugee gmail com
update mallsecurityconfig java
missing file header
remove qmuibottomsheet s overscrollmode
update pmsproductattributecategorydao java
use threadlocal withinitial to init threadlocal
ignoring some tests that hang in safari tp so that it does not respond anymore
make timezoneproperty compatible with jcasc
add support for getting return value of native windows threads
add license headers
lookup current call target after making a split decision
update pmsproductattributedao java
fix wrong logic in checking field changes
remove wrong invalidation of new class verison
dashmediasource ensure mpd location is applied in also case of http redirect
re add target_java_lang_compiler substitutions
turn processpropertiessupport default methods into regular interface methods
sulong use getter outside of annotations
add neverpartofcompilation assert
test update test testanonymousinline
update pmsproductattributevaluedao java
fix comment formatting
init value for grantedcols
remove unused code
replace deprecated methods in paramutil
array perse fix
update pmsproductcategoryattributerelationdao java
consistent assertions for supplier based bean definition tests
greenplum table cache fix use it always
support pre initializing identityhashcodesupport
mute timeserieslifecycleactionsit testhistoryiswrittenwithfailure signed off by mark vieira portugee gmail com
improved the performance of action updating at the expense of possibly doing the update twice
treat the accessor as a compilation constant
fix focus getting stuck by cycling within the same list item
intercept activitynotfoundexception for action_manage_overlay_permission
set correct hardcoded android min sdk version
cleanup the scanned objects cache
sulong fix debugger test style issues
sulong adopt hostallocatednativepointer
merge branch master of github com didi doraemonkit
make sure to adopt a replaced direct call node in ast to make it visible to instrumentation api
insert transfer to interpreter on direct call node updates
update pmsproductcategorydao java
add missing specialization for writing a long to memory typed as pointer
don t convert foreign objects eagerly to native in toi64
update core src main java hudson extensionlist java co authored by oleg nenashev o v nenashev gmail com
http header names are case insensitive
change to using textutils
clean up windowsphysicalmemorysupportimpl
hotfix network fix the comments error in nettymessage
fix dml revert wrong fix
export truffle to classloaders with impls of defaultexportprovider
update pmsproductfullreductiondao java
polish containsany of collectionutils by reusing findfirstmatch
remove pointless override of getscaletype
fix stamp of unsafecompareandswapnode
no log level check unless string concatenation is following
add groupie library to the credit list in the about page
update pmsproductladderdao java
add javadoc for the categories
ci correct spacing
ci use proper modifier order
ci bump copyright dates
ci combine if statement and fix formatting
update pmsproductvertifyrecorddao java
update qmuinormalpopup java fix qmuinormalpopup
ci bump copyright dates
ci combine if statement per pmd sonar
ci fix single line javadocs to multi line javadocs
ci bump copyright dates
dump ccompilerinvoker only when needed
update pmsproductvertifyrecorddao java
重写主流程 代码更加优雅 测试完毕
重写主流程 代码更加优雅 测试完毕
重写主流程 代码更加优雅 测试完毕
remove unused import
avoid modifying parameter
update comments for dash update the comment to indicate the dash equivalent use case for the exception
only replace existing deopt entry for merge nodes
optimize inaccurate judgment will lead to npe
fix a typo
update media timestamp regex to allow negative values
node cost dump util use boot class path
use binaryparserexception instead of runtimeexception
rename intconstantoffset to byteconstantoffset
update omsorderoperatehistorydao java
connect job fix direct execute mode
s throwonerror notfoundisfatal g
fix boot library path on windows
fix checkstyle warnings
add directive to run live test
ignore negative bci instanceof
fix buildimagemojo javadoc
using try catch and recursion add try catch and recursion to optimize the code
add readglobalindex executionstate state
print custom storeconstantspolicy
fix peek4 and peek8 performance
update smscoupondao java
hotfix add javadoc to registeredrpcconnection tryreconnect
hotfix minor cleanups in jobleaderservice
delete unused method in configurationclassparser
add inspur k db database support
hotfix tests fix exception in autoclosableprocess
add public default_max_size
update smscouponproductcategoryrelationdao java
bugfix use wrong ref
drop extra check
add deprecated method for compatible
merge remote tracking branch origin master
hotfix make testbulkwriterfactory publicly available for other test modules
class registry cannot be null
remove leftover for dns rebind attack on websockets
adjust the documentation to reflect the security impact of inspect path
add documentation for secureinspectorpathgenerator
replaced unicode dashes by ascii dashes
fix a typo
fix style issues field shadowed by a local variable
inspect path documentation adjustment the documentation is closer to the documentation on the website
fix warnings in tests
add documentation for dns rebind protection principles
avoid lock on staticobject instance
reduce attachmentciphertest flakiness
remove system out println from launcher
remove unused imports
update smscouponproductrelationdao java
fix nohttp violation
update decimaltobinary java close ressource leak scanner input
update cycles java closing scanner in
update primmst java remove unused import java lang
update cycles java remove unused member variable finalcycles
update main java fix bug program is causing a nosuchelementexception
remove some sysout debugging
update redblackbst java closing scanner scan
remove unnecessary check in convertingencoderdecodersupport
fix volatilereadnode simplify to simplify only when lastlocationaccess is set
fix litemall wx api 解决潜在的警告信息
model selector fix lkeep focus
java we can t use bazel runfiles root as project root on windows
remove unused imports
getstaticmethodid throws nosuchmethoderror for primitive classes
java unignoring a test implemented in chrome
merge branch master of https github com linlinjava litemall
java restoring thread interruption status before rethrowing an exception
java deleting unused imports
revert license header
fix a typo ci skip
turn comment into description for assertion
fix gc derived pointers
set proxybeansmethod to false when no method is called
improve javadoc in configuration
fix test failure in security master
improve performance of stringutils cleanpath
fix dumb mistake
mute blobstoreincrementalityit testincrementalbehavioronprimaryfailover
feat broker prevent blocked by group transfer service
remove unused code
don t omitframe setup when preserveframepointer is set
remove unused import
fix checkstyle issue
add name to error message
always query trufflelocator
return empty scopes even if env is null
sulong fail if llvmbitcodelibraryfunctions does not find a method
sulong fail if a library specified on command line is not add
sulong make loadc libraries off by default
sulong resolverenamedsymbols fail if symbol not found
sulong libraries specified on the command line might be already loaded via dependencies doing so should not fail
sulong fix performance warnings
sulong inject libsulong if we directly load libc or libc abi
don t query loaders when running in native image
fix integer overflow
seek duration should not be rounded up when opening the settings even if inexact seek is disabled
fix wildcard imports silly intellij config is being overriden at the moment
themehelper fix the the typo
update eladmin system src main java me zhengjie modules security security tokenfilter java 修复securityproperties参数配置自定义令牌前缀bug 既然允许自定义令牌前缀 就不应该固定截取七位
fix android q storage
hotfix tests remove dead code
move optionscommand to basic package
remove unused methods
hotfix add int type to job_manager_heap_memory_mb option
hotfix remove unused method in taskexecutorprocessutils
s javax annotation checkreturnvalue edu umd cs findbugs annotations checkreturnvalue g
s javax annotation overridingmethodsmustinvokesuper edu umd cs findbugs annotations overridemustinvoke g
fix typo in beandefinitionparserdelegate
defend against integer overflow in bitmap2d constructor
string values are read as ids in setvalues command
sulong do not attempt to repeatedly add sulong native so
sulong reformat resolverenamedsymbols c support
sulong fix comment in cxxdemangler
sulong remove unnecessary compilationfinal from nficontextextensionnficontextextension
entity editor default folder switch fix
sulong comments regarding initialization ordering
sulong do not add native function that cannot be loaded into the symbol table
update spring beans src main java org springframework beans factory xml beandefinitionparserdelegate java
update romantointeger java declaring serialversionuid
remove invalid nonnull annotation on a ctor
s javax annotation concurrent immutable net jcip annotations immutable
hotfix jdbc rename postgrescatalogitcase to postgrescatalogtest
fix pe infinite recursion modify compilationfinal s on interpreter
remove unnecessary lines
made sure updates to comparisons are done on swing thread
fix typo in task name
fix zero or negative time interval not supported
remote maxinstance restriction on retrieveprofileavatarjob
don t send group info requests in response to group info requests
fix npe after call failure
do not print random binary data during websocketagentstest
fix checkstyle errors
hotfix jdbc add license header to postgrescatalogitcase
test fix beatreactortest
merge remote tracking branch upstream develop into develop
attempt to make mbeanclientinterceptortests more robust
avoid arrayindexoutofboundsexception if stack trace does not contain a reflective call element
hotfix tests rename variable
hotfix coordination exit early for empty collection
hotfix metrics do not log reporter config to prevent credentials leak
hotfix metrics prometheus add missing import
sulong silence currently invalid performance warning
fix issue where leave message wasn t displayed locally
don t refresh own profile if not registered
use the map s codec when getting the map signed off by chris eager christian turo com
don t show empty date popovers
support null secrets when converting form submissions
set httponly and secure on cookies to fix spotbugs issue
only set secure if request is secure
use correct class for getting a vector drawable in older apis
use fstconfiguration deriveconfiguration to preserve conftype signed off by chris eager christian turo com
update loginfilter java
ignoring a bad flaky test
folder node icon resolve fix
merge remote tracking branch origin master
rename method to getlastbci
merge branch master of github com redisson redisson
distinguish native and non native jni wrappers
add public methods
fix column icons blinking on refresh reorder
fix declared patchcontext was not found when generic type argument was used with ecj
fix blacklisted method
fix test was making wrong assertions
fix import after rebase
check file name only if needed
fix issue where leave messages were pending forever
oracle auth backward compatibility fix
remove unthrown exception
make language agnostic inlining work when inlining is disabled
remove native image annotation from symbolicsnippetencoder
fix issue with some notifications linking to the wrong conversation
improve database migration sql statement
use delete annotation to disallow symbolicsnippetencoder in libgraal
fix checkstyle issue for servicerestmetadatastorage java
verify both agent cripts have been closed
fix npe when unknown request id is sent for clear breakpoints event
remove no longer necessary targetapi
adapt to the new getfieldvalue api
minor cleanup for code merged into androidinput from androidinputthreeplus
bugfix setbackgroundalpha has a side effect to global resource
bugfix the repeated value
add verbose timer for post gc reference processing
sulong use the isintrinsicglobalvariable check of the global symbol instead of going through the linkage itself
simplify check on ccompilerinvoker singleton
update javadoc for abstractlistenerwriteprocessor write co authored by rossen stoyanchev rstoyanchev pivotal io
ensure logprefix is not null or blank in abstractlistenerwriteprocessor
parameter bind dialog check params count
make select use trimmed text when doing text matching
testing case insensitively check tag names
testing only check one flag to see if we want legacy edge driver
java deleting unused imports
java add a private constructor to prevent instantiating a utility class
fix throw exception log describe
minor edit fix double spaces
change the parent class
eliminate unused killing begin nodes during virtualization
log compilation failures with tracecompilation and tracecompilationdetails flags
merge remote tracking branch origin devel into devel
disale tree expand for connection dialog
mysql system catalogs
fix integer overflows available method in chunkfileinputstream java free ahead space calculation in circularfilewriter java
mark requestlistener onloadfailed model parameter as nullable
update copyright year of changed files
reduce log level on notice about optional extensions
hotfix remove redundant check for queryablestateintegrationtest
avoid capturing the exception in the node just capture the error message
document instanceof filter
fix redissonbloomfilter throws npe
save changes handler fix
revert change to pathpattern javadoc
hotfix coordination add sanity checks to clusterpartitionreportentry
update serialization version after backport
fix unittest so it catches extra membar and fix wrong condition that causes extra membar
fix postcard addflags bug
restrict newly introduced api
hotfix metrics extend metricreporterfactory javadocs
avoid reflective instantiation for lambdas on svm
fix option parsing errors must be converted to engine errors
active query cancel fix async mode timeout
revert some more changes
bug 修复queryhelp getpredicate查询会产生多余的join
avoid potential race condition in attachment uploads
prevent empty message processing
prefer profile photo over system contact photo
when closing error handling can no longer the context to enter
simplify debugging of polyglot engine exception by initializing the cause
fix context should not be entered if closed or invalid
improve documentation for polyglotengineexception
check if already running before remove watched
native image lower safepoint node late
default lowering comment box canonicalization
fixes after rebase
array intrinsification make high level node a state split
snippet framestate merge assignment more debug messages
fixes after rebase
style fix the pmd rule
update reflectioncachesourcecreator java replace printing types to standard out through logging with log level info
return false not throw exception
fix datasource description
make null sortable
omit vm config value check for dev jvmci binaries
java preventing possible nullpointerexception
java this method never throws this exception
fix compilation error after merge
code style fix
java restore thread interruption state after handling interruptedexception
java deleting unnecessary check
hotfix metrics javadocs fix typo
fix test failed
bug fix for isuseunfairlock
polish javadoc for transactionmanagementconfigurer
fixup refactor reanme calldirect to calldirectorinlined
add target call graph for key optimizedcalltarget methods
merge branch master of https github com alibaba druid
merge branch master of https github com alibaba druid
fix invalid assertion
increase timeout threads keep alive time
increase reliability of exception handling when context is closed
change pea unit test to work if the merge is eliminated
java setting process to null only if it dies prematurely
java add debug output to find out why assume does not work on travis
cleanup paths to fix tests running on windows
remove windows line endings when parsing index
attempt to fix failing windows test
java more debug output to investigate travis environment variables
attempt to fix failing windows test
java oh i ve mixed up system properties with env variables
remove unused methods
add missing null check
add type parameter
remove deprecated use of character
remove unused variable
remove unused method
remove unused variable
remove unnecessary suppresswarning
fix trailing spaces
request cancellable and log are nullable cancellable because of tests
don t use assert class from loader remove the use of assert since it s unavailable that early
fix missing subtitle addition
verifydebugusage now reflects refactoring
make sure replacements contains the child invoke
manipulate the msg in interpreter mode
fixup refactor rename and move a few constants
connection recover on isolated context open
add import for previous changes
correct init of expanded count in default policy
hotfix fix logical type generalization for approximate types
mute samlserviceproviderdocumenttests teststreamroundtripwithallfields
mute modelloadingservicetests testmaxcachedlimitreached
clean up playwhenready
resolving javadoc imports and other lints
修复不存在账号登录时 userservice findbyname抛出异常 而loaduserbyusername未将其捕获处理的bug
update loginfilter java
honor overriden accepptheaderlocalecontextresolver getdefaultlocale
cachekeyfactory is not used
remove apparently unused slave labels
fix visit loaded source sections
don t clone instrumentation wrappers
java waiting for process finish should be interruptible
java small performance improvement
java deleting unused imports
java modifier private for enum constructor is redundant
java minor performance improvement
add barrier before releasing lock in fast path
ignore static fields in llvm printast
mysql indexes overload
connection page title update fix
java linter recommends replacing n with n in string format
java deprecating okhttpclient prior to deletion
fix checkstyle violations in mediasourcemanager
hide tabs when a video is age restricted
don t break navigation if player views other than controls are focused
调度中心与执行器通讯规范为双向restful 方便跨语言 以及第三方执行器实现 通讯组件xxl rpc方案调整为jetty gson方案
java deleting redundant brackets
reverted the file name
fix junit test case
corrected confusing shouldasktoanalyze description
grid throwing a more specific exception to show the essence of the problem
changed linkedhashmap to map interface
varius get cases
java deleting an unnecessary sleep tests should work without it
java deleting another unnecessary sleep tests should work without it
java refactoring code simplification
java add todo comments to visualize unimplemented tests in ide
fast paths for long and native pointer in llvmaddressequalsnode
fix some typos within comments
fix commandbatchservice throws npe
mute eqlcancellationit testcancellation
mute searchablesnapshotsintegtests testcreateandrestoresearchablesnapshot
mute inferenceingestit testpipelineingest
generic connection page layout fix
add fast paths for long and native pointer in llvmpointercomparenode
hotfix coordination log if taskexecutor establishresourcemanagerconnection fails
unconfiguredschemametadata not config is npe
correct logs information in kafkapartitiondiscoverer
fix missing boundary in library
support oceanbase exceptionsorter signed off by hongwei yi hongweiyi hotmail com
update umsadminpermissionrelationdao java
don t check for a specific gc for excludefromreferencemap
use referent instead of rawreferent
remove author comment tag in parseastnode
add data to version graalvm
update session rmap after changing id signed off by chris eager christian turo com
fix racy test
java fix formatting no functional changes
add missing serialversionuid
java minor performance improvement no need to load options if section name is empty
java refactoring to shorten code
java deleting unnecessary check looping over an empty collection is ok
java using the same name for a filed and a local variable is ambiguous
java re interrupting the thread after handling interruptedexception
java minor refactoring to reduce code duplication
fix comment about when this is set
hotfix runtime remove unused method executionjobvertex generatedebugstring
hotfix rocksdb fix exception message
made ishiddenclass optional
grid vertical scrolling
remove duplicate check if cookies are already present
update umsroledao java
mute classificationit testsetupgrademode_existingtaskgetsunassigned
mute modelloadingservicetests testmaxcachedlimitreached
remove an unused method on leafdoclookup
log output fix
hotfix sql add name for exector
skip the loop
also use value comparison for maps
transactions don t work
hotfix javadocs fix typo
allow intentional hiding of field names
explain why using newer profiled types is ok and what that relies on
expand comment about profiling fields
the profiling assumptions need to be volatile otherwise no order can be guaranteed
fix check in injectreturnvalueprofile it should read the assumption first
add comment about why transfertointerpreter is used
change sorts shellsort java
update umsroledao java
use package private visibility to avoid synthetic methods to access fields
update shellsort java
whitelist biginteger signum for pe just a field read
add saving subtitle track name for matroskaextractor
fix missing call_profiled in polyglotvalue that would trigger an assertionerror since that would end up in call callindirect stopprofilingarguments
java using stringtokenizer looks old fashioned
remove unessesary float
hotfix fix active task process ratio metric recording
permissionsfeature should not be a automaticfeature
hotfix runtime code clean ups in activeresourcemanagerfactory
hotfix fix perjobminiclusterfactory does not properly calculate numslotspertaskmanager
java deleting warning word from a warning message logger adds it automatically
hotfix network move uncritical code out of lock in remoteinputchannel onbuffer
ruleschemametadata should be final
fix checkstyle warnings
avoid object equals add final equals hashcode implementations
test that testconstructor can be used as a meta annotation
fix colorfilte colorfilter path issue has been fix for androi
invalidate on out of bounds memory accesses
profile to bounds checks
java actually ignoring an exception as stated in the comment
merge remote tracking branch origin devel into devel
hotfix runtime fix import order in pipelinedregioncomputeutil
allow condition inputs in conditional conversion
update umsroledao java
java no need to declare uncheckedioexception in a method signature it s unchecked
java deleting unused imports
java using try with resources to close the stream
minor spotbugs performance improvement
text hint render fix
remove outdated comment
enqueue on long click on background popup in channel
update cmsprefrenceareaproductrelationdao java
first commit in new fork
add getter and setter
also intrinsify float and double variant of llvm round intrinsic
disable failing test temporarily
apply nuller information from loop header to loop node
use constant for error message
update cmssubjectproductrelationdao java
add mising file update
skip tests that do not work with agnostic inlining when it s enabled
fix auth mechanism activation by driver
hotfix fix the code style in bashjavautilsitcase
put system currenttimemillis behind boundary
hotfix add rpcutils inf_duration
java deleting unused private constructor
java minor performance improvement
fix management context pointer null checks
java fix a flaky firefox test it can fire more events
reduce raw stacktrace dumping
remove unused method sourcelookup extractvalue
remove duplicate line
add logging when context null in onpreftreeclick
optimize code remove not used global variable
merge remote tracking branch origin devel into devel
fix returnbinding save stack
document purpose of accessverifier accessadvisor
enable loadclass instrumentation by default
clear classloader parallellockmap
database transfer progress messages fix
don t materialize source sections when asking for isinstrumentable
update pmsproductdao java
make sure inaccessible javascript is reported as truffleexception
fix clear screen in test
apache license comment add
check menu item visibility when calculating menu size
remove long press action in settings
account for deleted conversations in profile key send job
interpret non present message bodies as null
bugfix error message
minor javadoc improvements signed off by paulo lopes pmlopes gmail com
mute modelloadingservicetests testmaxcachedlimitreached
sharding metadata refresh on listen change
merge remote tracking branch origin developing into developing
fail with descriptive error if native access is not allowed when spawning espresso
fix native image buffer overflow on windows using chinese charset
avoid deprecation warning in defaultrsocketrequesterbuildertests
dismiss group participant list on contact click
fix call termination when muting before call is connected
fix button ripples in videodetailfragment
update pin change strings
android fix repeat intercept
rename sameness to identity semantics in documentationn
refactor fix code stype
update pmsskustockdao java
improve logging for pinstate
oracle connection page fix resize for long custom urls
hotfix fix checkstyle violations in standalonehaservices
hotfix remove unused resource_manager_rpc_endpoint_name from standalonehaservices
mute modelloadingservicetests testmaxcachedlimitreached
java deleting unused imports
java add a little bit of assertj magic to avoid type casting
java deleting redundant junit s runwith annotation
attempt to fix ci
backport attempt to fix ci
adapt code after backport
turn printbackendcfg off by default
remove duplicate shutdown producer
fix attachment upload bug
hotfix filesystems remove unused stopwatch
hotfix runtime throw exception from testingschedulingpipelinedregion getvertex for unknown vertices
修改 sheet 事件未调用的bug
java deleting redundant explicit type parameter
geo fix checkstyle failure remove unused import from direct push bugfix
java deleting redundant fixture initialization
fix style and comment
simplify rootnode initializelock
bumped required jvmci version to get changes add missing vm config values
update asmutils java
update methoddeclarationbinding java
update smscoupondao java
java minor code cleanup
java reducing code duplication by reusing existing method to shut down test environment
java using native java facilities to create a shallow copy of a map
java disabling a test on travis because using ping is not allowed there
hotfix yarn code clean ups in registerapplicationmasterresponsereflectortest
hotfix yarn test avoid using mockito for taskexecutorgateway in yarnresourcemanagertest
update progress bar on fast forward rewind
rid pointless wrapping with bufferedinputstream from class reading
add trailing slash to fix the relative path bug
remove unnecessary method call
modify log output
java oops wrong copy parse
generic connection page fix refresh url for non custom connections
hashmap constructor choose more appropriate initialcapacity
replaced for loop with enhanced for
print more details when processimpl error
fix fix double lock
update smscouponproductcategoryrelationdao java
fix deobf resolve npe when package is empty if rename is disabled
java changing a variable name to follow java conventions
makes loop cleanupwindows and cleanup protected
hotfix remove unused total_flink_memory_mb field
hotfix let taskexecutoroperatoreventhandlingtest extend testlogger
fix crashes on rotation by checking for null
hotfix fix processmemoryutilstestbase testexceptionshouldcontainrequiredconfigoptions to run also for tm
hotfix remove taskmanager from memorybackwardscompatibilityutils getlegacytaskmanagerheapmemoryifexplicitlyconfigured
make stringindexofcharconstantbindingtest amd64 specific
modified the cea608decoder constructor
replacing flush with collection emptylist
intercept substratetruffleruntime log via trufflefeature support
update smscouponproductrelationdao java
more precise return type for getlazylock
hotfix yarn code clean ups in workerspeccontainerresourceadapter
hotfix runtime code clean ups in resourcemanagerruntimeservices
update assertion message for callprofiled
fix threading issue with message resends
fix controllers to use post redirect get forms should use post redirect get pattern
pg object navigation hyperlinks fix when editor schema differs from default one
intrinsify methodhandleimpl iscompileconstant object
remove direct usage of vertxhandler from connections
fix duplicate line function vertxoptions tostring erroneously returns addressresolver field twice signed off by jan dolinar dolik rce gmail com
support classloader createorgetclassloadervaluemap
fix tests to check for forward to index
update prologue address
update smsflashpromotionproductrelationdao java
grid tests throws is not needed skip ci
show pin reminder for non reglock users
fix issue with group storage ids
fix textfield cursor not blinking
ensure timer removes its listener from the same gdx app instance
prevent exception from being swallowed in maven plugin adds root cause to mojoexecutionexception when an illegalaccesserror is thrown
add missing copyright header
hotfix runtime add sanity check to slotmanagerconfiguration
hotfix checkpointing use practical channelstatereader instead of no_op
blank line in the end file
update umsadminpermissionrelationdao java
mute failing tests in asyncsearchactionit
add removeduplicates function
mute cancellabletasksit testdonotwaitforcompletion
revise code style
add test annotation
fix code style
remove redundant supressions
hotfix javadocs fix typo
changes in build pom xml
refactoring code in com baeldung modelmapper package
remove unused locationonscreen variable
dbe configurator refactoring
dbe configurator refactoring
fix issue with pin verification
fix rejectedexecutionexception thrown by redisclient connectasync method during shutdown process
resolves cr comments
security utils additions md5
update umsadminrolerelationdao java
use cache for finalising the graph
hotfix add comments to filetype enum values
consumer changed as defaultconsumer
java using logger instead of printing to stdout
java fix internal links in javadocs
asm verity support print details
update umsroledao java
fill in further since todo comments
further since comments not picked up by tool
grid using streamsupport stream credits to barancev skip ci
is an unsafe character therefore do not use in a project name
these at since were improperly add
最近在阅读贵公司源码时候 本地代码工具扫描出几处performance defect iteration can be replaced with bulk collection addall call 于是有代码洁癖的我 厚着脸皮提个pr给到你们
update umsrolepermissionrelationdao java
only load truffle with nativeimageclassloader if building truffle images
make deprecated a meta label not shown in the regular label list
make the systemproperties api available to plugins
make parsing this safer
remove reference to protector from scrambler entirely unrelated
count with new sources being add during new sources notification
rename hiding variable
fix source execution bindings lock bug
fix source section with negative start line was breaking the jpsx emulator dynamically generated bytecode
change tiledmaptilelayer tilewidth height to an integer
implement search with search helper
hotfix set correct numiterations in shouldallowconcurrentaccesses
word types relax verification for svm
use getname instead of tostring
null safety with labels array
close library paths after use
older eclipse compatibility
use correct variable
update miniouploaddto java
fix messageconsumerimpl failure message signed off by thomas segismont tsegismont gmail com
initialize deliverystrategy with vert x instance rebase issue signed off by thomas segismont tsegismont gmail com
nodeinfo should get the actual port if options had wilcard port signed off by thomas segismont tsegismont gmail com
typo signed off by thomas segismont tsegismont gmail com
add setvertx method to deliverystrategy signed off by thomas segismont tsegismont gmail com
provide clustermanager with vertxinternal in practice most cluster managers use this api signed off by thomas segismont tsegismont gmail com
minor doc update signed off by thomas segismont tsegismont gmail com
remove unnecessary map operation signed off by thomas segismont tsegismont gmail com
use oncomplete instead of onsuccess onfailure signed off by thomas segismont tsegismont gmail com
remove unused var signed off by thomas segismont tsegismont gmail com
fix bug in defaultdeliverystrategy signed off by thomas segismont tsegismont gmail com
add unwrap method to clustermanagerdelegate signed off by thomas segismont tsegismont gmail com
replace sethandler calls with oncomplete signed off by thomas segismont tsegismont gmail com
failsafe when trying to connect to an unknown remote node signed off by thomas segismont tsegismont gmail com
typo signed off by thomas segismont tsegismont gmail com
registrationstream clarify threading model signed off by thomas segismont tsegismont gmail com
fix fakeclustermanager signed off by thomas segismont tsegismont gmail com
minor remove unused field signed off by thomas segismont tsegismont gmail com
dequeue waiters in batches signed off by thomas segismont tsegismont gmail com
avoid duplicating accessible nodes list it is an unmodifiable list already signed off by thomas segismont tsegismont gmail com
make sure vert x sysprops are cleared before running command tests signed off by thomas segismont tsegismont gmail com
fix possible npe in fakeclustermanager signed off by thomas segismont tsegismont gmail com
correct computation of prologue size
implementation notes for weightedroundrobinselector signed off by thomas segismont tsegismont gmail com
fix it tests that was not passing anymore due to using decodeexception instead of unsupportedoperationexception
remove system out println introduced by mistake
fix assertion in llvmloopdispatchnode
hotfix table common enable receiving isbounded in dynamictablesink
mute searchablesnapshotdirectorytests testindexsearcher
mute classificationit testdependentvariablecardinalitytoohighbutwithquerymakesitwithinrange
mute regressionit testtwojobswithsamerandomizeseedusesametrainingset
mute transformsurvivesupgradeit testtransformrollingupgrade
sulong remove looking up symbol with export
sulong look for target of alias only in the file scope
sulong add truffleboundary for nficontextextension
sulong add comment to initializationorder
return unavailable source sections if the line numbers are incorrect seen with dynamically generated bytecodes embedding a dummy linenumbertable attribute
java an attempt to fix a test in chrome on travis
don t include non sections in segment length
update omsmoneyinfoparam java
pg dump enquote schema table names
filters state update fix
llvmreadnode clean up
object editor form layout fix
fixes npe after metadata without configured changed
sharding proxy config and integrate metrics
tinker fix build failure
sharding proxy config and integrate metrics
tinker avoid odex checking on meizu devices since it would not generate odex when load dex first time
allow qmuiswipeviewholder to clear actions
pg procedure signature fix do not quote standard data type names
解决由于controllermethodscache类的getmethod方法中 由于replace target写死造成的修改contextpath后找不到api的异常
sharding proxy config and integrate metrics
update omsorderdetail java
make index implement intunaryoperator signed off by thomas segismont tsegismont gmail com
hotfix fix checkstyle violations in bloblibrarycachemanager
hotfix fix checkstyle violations in librarycachemanager
hotfix fix checkstyle violations in taskmanagerservicesbuilder
hotfix network add debug message marking end of output recovery for unaligned checkpoints
trust cluster manager for not modifying the event data after firing signed off by thomas segismont tsegismont gmail com
rename emptyselector to nullselector signed off by thomas segismont tsegismont gmail com
remove diamond operator
fix deliverystrategy doc there are now distinct methods for send vs publish signed off by thomas segismont tsegismont gmail com
add scanner in factorail program
spreadsheet update panels on value paste
avoid need for accessor
simplify regex expressions
add comment in front of main method
navigate to parent uploader channel from the uploader section on long click
add indentation in the main function statement block
merge remote tracking branch origin master conflicts eladmin system src main java me zhengjie modules mnt util scpclientutil java
fix logging of ui server auto stop fix indentation in test signed off by tamás fenyvesi tamas fenyvesi doknet hu
fix releasebytearrayelements signature
update test name
correction when no meta object is available
update omsorderreturnapplyresult java
deprecating jenkins model configuration
fix flaky loggingeventlistenertest
fix concurrentmodificationexception due to nested creation of isolate singletons
hotfix table common make scan and sink runtimeprovider interfaces public
remove unused import
merge remote tracking branch eclipse master
fix codacy issues
update omsreceiverinfoparam java
cleanup stderr output
tinker bugfix a serious typo
update pmsproductattributecategoryitem java
add unsupported testcase scenario
tinker adjust some logs
add final for shardingspheredatasource
add runtimecontext log
fill shadowrule shadowmappings
remove empty line
macos cli params escape fix
update pmsproductcategorywithchildrenitem java
add missing replacement of object array elements when building heap histogram data
make resources in image heap read only
fix dead lock in libgraal shutdown related to jmx bridge
singleton model by lazy load in metrics facade
hasmembers can always return true
hotfix network fix useless backlog value in bufferandavailability returned by remoteinputchannel getnextbuffer
do not hold a lock when calling svmtohotspotentrypoints signalregistrationrequest
resolve checkstyle problems
fix merge conflict
java deleting unused imports
java deleting unused private variable
java reusing an existing constant instead of a hardcoded string value
java add javadocs for httpclient factory methods
java as far as we don t apply filters to websockets it s safe to use base implementation
fix checkstyle violation
use string reference
java fix npe in reactor httpclient if the server returns a response with no content
update avltree java
update vampirenumber java
make sure to set the interrupt flag
value panel refresh on rsv rows navigation
fix check style
fix sending e mail from crash reporter
feat open jraft node metrics
fix fix code style
document why imageincludebuiltinmodules cannot be a hostedoption
polish javadoc for namematchtransactionattributesource
after rebase fixes
update pmsproductresult java
use release notes in message as it is more generic
sulong change getlanguagecontextextension to be private
sulong minor rename getlanguagecontextextensions
ensure you can t set null values in defaultvaluelivedata
fix crash in submitdebuglogactivity
add long press api
throw should not reach here exceptions on invalid jdwp class patterns
pinned columns fix struct columns
fix image heap permissions
fix pmd problem
fix fix code style
update productattrinfo java
fix start submitdebuglog from registration and passphrase prompt
ensure that the unknown uuid does not create an entry
reduce recipient dirty state logging verbosity
prevent some ioexceptions when past the end of stream
fix message receive timestamps on media messages
correct access control for mms groups
allow auto download for users you ve shared your profile with
enable speaker when initiating a video call
fix itemgroupmixintest compilation
expand tappable area in header
raised program version due to the revised memory support of overlay mapped blocks and byte mapping schemes
fix svmtest for tests that expect a runtime exception
add space for timezones message timezonesby default
increase bintray timeout
fix clusteredeventbustest testimmediateunregistration signed off by thomas segismont tsegismont gmail com
oracle remove synonyms from schema children too slow md read
enable note to self as recipient in share activity
live group membership count in conversation
fix nullpointerexception when trying to launch inviteactivity
ml fix typo in comment
fix style check error
keep a threadlocal for the simpleformatter
fix controls are remove when remote video is disabled
minor change to setlabelprimarycmd
unlink method calls deleteasync inside instead of unlinkasync signed off by aayushya vajpayee aayushyavajpayee gmail com
sql server connection page fix graphics is disposed
update smscouponparam java
hotfix fix table walkthrough due to remove tablesource tablesink registration
hotfix checkpointing fix the invalid argument in constuctor of checkpointcoordinatorconfiguration
updated java comment
modified mvcvoterconfig controller path
update copyright years
unit test renamed
hotfix tests make fscheckpointstateoutputstreamtest work on windows os
hotfix tests remove unused temp folder in checkpointmetadataloadingtest
ci sort imports
ci remove unused import
remove unused code
update smsflashpromotionproduct java
hotfix table common add validateexcept to factoryutil
better job timeout timer name
update license year
hotfix fix the access modifiers in workerspeccontainerresourceadapter
hotfix remove intellij noinspection directives from memorymanager
java fix devmode detection in idea
fix fix future join blocking thread
revise java doc
sulong minor formatting
regression in interpretercallbenchmark directcall
update smsflashpromotionsessiondetail java
fix junit tests
fix compilation error
disable view prefetching for now
fixup consolidate call ui visibility selection logic
add necessary empty line
style fix code style
update package name
remove of sql
prevent failure on windows
grid correct reporting of purpose of class when instantiating the session map
grid add missing event bus role to message bus command
increase memory available to netty client for websocket frames
default to logging warnings when the websocket has an error
tracing remove some noise
check for recursion depth and depth before checking forced inline
navigator nodes search fix
fix checkstyle errors
revise load instance state
update umsmenunode java
ignore test case for spring boot
remove unused code
update selectionsort java used compareto used a local swap method
substratenewinstancenode has been remove newinstancenode is now sufficient
provided credentials read fix
log successful pin setting
fix possible data source invalidation loop
remove invalid assertion
provided credentials read fix
hotfix table planner blink fix exception for invalid function signature
ignore for now this test as it does not reliably pass
check classloader before enhance
use closehooks directly instead of going through the context
make output of info command more readable
remove invalid getter
revise test class be final
update umspermissionnode java
sulong fix npe in vectortype exception message
sulong vectortype should not call tostring of its elements for exception message
clean object in objectstack
search cl exe version string so that it works for non english setups
gis viewer tooltips generation error handle
java fix edgehtml driver initialization in tests
properly set profile key update flag
hotfix core add remove missing incorrect serialversionuid fields in timestamp assigner classes
grid autoservice annotation out to avoid duplicate options for now
fix increase hashslot value
java use logger instead of printing to stderr
hotfix core add open interval in migrationversion
hotfix fix directory creation in typeserializerupgradetestbase
remove commented out code
add agentutils install method
merge branch master of github com alibaba arthas
update osscallbackparam java
handle all pclath modification cases
replace hardcoded white drawable with themed one
fix mainfragment tab icons did not follow theme color
refactor removes the http forward for the configuration module
project explorer filter
merge remote tracking branch origin devel into devel
don t create identity change methods for brand new contacts
ensure we upload the profile after a pin restore
fix the style
remove custom node count hack for old inlining
revise heart beat response
implement unsafe getloadaverage
fix logging assertion
fix incorrect argument count
hotfix qs fix logging of exception in requestwritelistener log stacktrace of exception instead of only the message
hotfix tests make clienttest extend testlogger
fix usage of hashcode in objectscanner
insight roots directive requires the language to provide support for rootbodytag
update bucketsort java
improved protobuf support
merge branch master of https github com alibaba fastjson
don t insert identity verification message for the initial restore
java fix javadocs structure issues
java deleting unused imports
fix fix notify ut
add exception throwing to withdecimalformatpattern method
don t store loaders statically for reclaiming change logging level of purge info classloader count no longer volatile
avoid using latch use the thread join method instead signed off by paulo lopes pmlopes gmail com
avoid breaking on unknown flow types
include unsafepartition flow types into the dump
add instanceof checks before calling earlycanonicalization
make clilink use read permission
add access to displayable function name
fix primary checkstyle error in javadoc comment
select a new namesrv when namesrvaddrchoosed not in new addr list
sql editor dispose fix prefs save
android 完成函数调用栈n级调用 用户自定义
enable sulongengineoption for interop tests
mute datehistogramaggregatortests testassubagg
add extra vm options to make reservedstackaccesstest pass
close shellserver and sessionmanager when shutdown
java remove misleading serializable interface from capabilities because they are not serializable
java shortening code a little bit
java oh what a silly mistake
clearly indicate the misconfigured class name in the gradle build message
fix checkstyle issues
mute recoveryrequesttrackertests testidempotencyisenforced
hotfix runtime fix typo
fix findbugs warning about possible npe
check for llvm in substrate lowering providers
fix deprecation link
reverse condition to clarify when it can apply
change test class name
hotfix core add a constant for the special value no_timestamp
processor uses current year to create copyright headers
fix the datasource assertion issue
java avoiding use of raw types
java using the provided output streams instead of the standard ones
java fix potential npe
fix correct skip size for sparse switch payload
fix comment typo
fix checkstyle issues
fix spelling error
only report the stub signature in stub tostring
sulong structuretype isnamed
improve efficiency by replacing llvmlanguage getcontext
introduce pinnedobjectsupport ispinned
change http client implementation unified apachehttpclient
remove unneeded imports
android fix support dokit ui style
fix instantiation of writenode
update core src main java hudson slaves slavecomputer java co authored by jesse glick jglick cloudbees com
update pmsproductattributecategoryitem java
remove extra lines
update nodefield documentation
ensure that the alternativelibcfeature is only active on linux
fix serverstatus api jackson error
fix deserialize error for raft beat
fix deserialize error for raft beat
fix deserialize error for raft beat
generic meta model table load fix null check
fix fix startup failure
fix fix unit tests that do not conform to pmd
add missing truffleboundary in sl
alphabetical member order
add a network constraint to remoteconfigrefreshjob
clear sticky webrtcviewmodel events when initiating a new call
using accessors for balance and stamp
reduce duration but increased forks of jmh micros
remove num thread restriction in arrayallocationbenchmark
extra warmup in arraylistbenchmark is not necessary
reduce number of tested sizes in arrayallocationbenchmark
on the next line
open the correct kiosk
hotfix checkpointing improve exception in case coordinator state ack fails
hotfix checkpointing beautify stack trace by stripping wrapping completionexceptions
hotfix runtime fox log message for web log file to only pring config key and not deprecated keys
sql completion processor npe fix
changing the test class name from integrationunittest to integrationtest
moved test class closer to class and add a test
remove accidently add import
for checkcast copy supercheckoffset needs to come from destelemklass
script config fix
add class file license
add class file license
merge pom xml
minor test improvement
reuse the server field for some tests in nettest
update pmsproductcategorywithchildrenitem java
note deprecation of classicpluginstrategy useantclassloader
grid avoid having duplicate flags
grid session map status should be value wrapped
use the right property in test
let some time to the server close to happen before starting the new server
add the binary examples
update litemall admin api src main java org linlinjava litemall admin config shiroconfig java 优化接口文档显示和请求
refactor add log printing
handle only permanent bailout exceptions for bailout state
minor improvement of nodeinfotest that will fail with ignite
minor improvement of eventbusregistrationracetest
fix wave form ioexception thread issue
clear system properties when the test tears down
remove the long running test case
be more explicit with the id we use for account updates
fix typos in logging for remote delete
also use addskipcheckinginfo if getccompilerinfo returns null
fix native image build problem
hotfix rename taskslottabletest into taskslottableimpltest
hotfix fix checkstyle violation in taskslottabletest
hotfix add taskslottableimpltest testallocatedslottimeout
introducing integer cache
fix junit errors caused by last week s listingmodel change
pass along allowassumptions flag from getsubstitution to getmethodsubstitution
fix expression stack for methodhandle intrinsifications
allow methods defined in varhandle itself
fix broken locals for static methods and add a test that catches the issue
improve help message
fix fix code style
add final to static fields where possible truffle
fix typo in node
made edit profile save button move with the keyboard
fix issue where rail wasn t showing in some situations
smoother blur rendering
java fix a broken test
initialize assumptions call when creating llvmlanguage instance
add missing license header
don t use useless getter that should be remove
cop paster copyright notices remove
conflict handler correction
remove unused import
fix unused imports
merge branch develop of github com alibaba nacos into develop
open up jdk internal org objectweb asm so that it can be used by com oracle svm hosted
java add back keepalive that was remove by accident
if possible to cancel future reset the task
robustify scanning for cl exe version info
add domainobjectchangerecord event type information
tool extra options
virtual entity editor support visibility fix for datasources with enabled ri only
remove listeners when detaching conversation item views
update copyright year of changed files
preserve scroll position in message details on update
don t show conversation settings for groups
fix conversation item and data source memory leaks
fix possible crash with null thread body
hi here addall call can be replaced with parametrized constructor call performance will be better
remove group members button
update pmsproductresult java
fix fix task executable judgment logic problems
make custom notification dialog dismiss itself on up press
fix search crash adapter array index out of bounds
restore mute in conversation menu
shorten logging in conversationactivity
change jobid field type to int
bugfix add protection
bugfix add protection
bugfix copy paste error
address comments on pr
default to using mysql s longer connection timeout var
sulong remove unused verifycyclefree method from type
sulong make type getrawtypearray public
sulong remove unused code from llvmfunctioncode intrinsic
code indentation fix
fix exception javadoc
also assert resource exhausted in allocation reporter test
make unhandled exceptions easier to distinguish
ensure that no floating reads are scheduled into c to java transition
adjust documentation for nodeintrinsic
fix signature for intrinsify
fix rtimeseries destroy method doesn t work
update parseinteger java fix error for empty string
hotfix checkpointing add visiblefortesting annotation for related methods
fix checkstyle violation
hotfix tests add missing import
hotfix testing remove the unused import from checkpointbarrierunalignertest
sulong llvmtoolchainnode should not return plain null but an llvmpointer
polish javadoc for propertysource
reinstate highlighter under drawing menu
fix issue where invalid pagedlist objects were passed to conversationadapter
delete dead code in mockserverhttprequest
fix bad specialization name
clarify initialization in javadoc of instantiate message
minor fix in llvm test does not need limit expression
add test for caching context property
add lwjglgraphics resetdeltatime
remove system properties from unit tests
prevent highlighter opacity affecting blur tool
tregex style fixes
a power of two size for hashmap
remove unnecessary type syntax conversion
hotfix correct the default path of nvidia gpu discovery
sulong add more javadoc to typearray
hotfix cli update the help message of the generic cli
fix navigate up behavior for conversations
show group participants menu item on a mms group
improve lazy loaded singleton example
fix issue with typing indicators in blocked groups
display unknown group for groups with no name
allowed user to copy video title to clipboard from detail screen
make fid cache size match the number of functions
fix checkstyle issue
hotfix table common fix tableschemautils getphysicalschema should keep the original constraint name
java handle if logging is not available on the hub
warning instead of exception if no root found
print normalized samples as well
new warmup definition
fix missing pads field assignment on atlasregion copy constructor
update productattrinfo java
builtin llvm prefetch implemented
adds rt jar back to the class path
hotfix runtime set root cause to pending request released exception
hotfix fix checkstyle violations in allocatedslot
properly throw unregistereduserexception in signalservicepipe
editor dispose fix
remove unnecessary setlongclickable
inverted if else statement
made clipboardmanager final
reset bootloader class_loader_value_map
update copyright messages
nullability warning in interoplibrary javadoc
fix line break
hotfix improve exception message for parsing kryo serializer classes from config
prevent extra buffer copies when parsing did not extract anything signed off by thomas cataldo thomas cataldo blue mind net
update javadoc for assert regarding guava preconditions
better format for simple output
prevented unnecessary test output
improved mapdeserializer compatible
fix invalid import
hotfix add nullable annotation to futureutils getwithoutexception
check package name for extendedsocketoptions
update smscouponparam java
hotfix yarn code clean up in yarnresourcemanager
remove test for current localisation as it was failing at prod due to different systems
update umsadminloginparam java
add license for externaldatasourceproperties
fix word spelling errors
fix checkstyle error
code format fix
suppress warnings in responsecreatorstests
change to final
use isassignablefrom to check if effect is matched
when looking up tostring method search for a method returning string add tests calling tostring on interop objects
fix the check if integer or long can be represented by float
since asfloat for long does conversion check instead of range check asdouble should do the same
minor in flinkkafkainternalproducer add javadoc to internal enqueuenewpartitions
remove unused constructor for controlflowexception
fix missing transfertointerpreterandinvaldiate
java fix static resources to allow running tests from idea
statistics renderer fix check total size presence
refactor method name
change fitsinfloat fitsindouble to follow the changes in asfloat asdouble
hotfix table common relax literal casting check in adaptedcallcontext
hotfix runtime narrow down the access scope of slotexecutionvertexassignment
hotfix runtime narrow down the access scope of defaultexecutionslotallocator
tinker do not hack mh in app oncreate again to avoid potential dead loop
tinker add log before suicide in appinfochangeblocker
tinker don t throw exception when performsecondarydexopt return false
shouldnotreachhere should be an assertionerror for easier debugging
fix vm classpath documentation
excelproperty 新增 order 用于排序
excelproperty 新增 order 用于排序
merge remote tracking branch origin devel into devel
cannot use a singleton for unsupportedmessageexception yet
java using try with resources to automatically close prepared statements
java deleting an unused private field
java using try with resources to automatically close prepared statements forgot to change this one in revious commits
java deleting unused imports
prevent illegalstateexception on recipient bottom sheet
don t prefetch views for the conversation list
ensure speaker off at start of any call
use recipient display name in recipient bottom sheet
fix localvarsbinding type
read elimination use sysid hashcode instead of node hash code to enable caching of not constructed nodes yet
use a list to collect unmarked multiguardnodes
improve error output for fails in languagespitest
fix lazy may be null when internal file system is used
if canon allow same direction being two loop ends to the same loop with the same phi value
fix improper tinting on screens when using fallbackphoto
fixes enqueue resuming paused videos
remove redundant stop calls from livereloadservertests
fix transient failure of deepunrollingtest increase margin a bit print timing in case of failure
suppress warnings in gradle build
default value in errorstateholder messageholder might leak
fix confusing error message when using vm x options
use contextclassloader when loading a class
fix serialization version in autofollowmetadata this commit fixes the serialization version on the settings in the auto follow pattern metadata
stop back transition jank from avatar viewer to settings
make conversation footer always show
fixes crash when a file is deleted then redownloaded
tinker unify author s name
read boolean s value through a field instead of calling a method
hotfix security remove never thrown exception in method signature
standalone always wants the local node
grid remove nodes from the distributor once draining is complete
tregex escape regex source in compilation unit names
matchablehandlermapping is backwards compatible
add the code samples
fix jndi test
fix jndi test
add issue reference
optimized code for ldapuserservice class avoid creating useless objects
note to future person
check article code matches github
memory update state after changing overlay property
remove override of remove method
task view listener fix
update ossconfig java
hotfix annotate the defaultclusterclientserviceloader as internal
simple libgraal bridge
hotfix javadoc fix typo in checkpointconfig enableunalignedcheckpoints java doc
jnienv newstring correctly takes values from the given buffer
do array clone check in access checks
hotfix remove unused slotpoolimpl get slotrequestid
hotfix fix checkstyile violations in bashjavautilstest
hotfix fix checkstyle violations in jobmanagerprocessutilstest
hotfix table common fix typo in encodingformatfactory
hotfix core fix typo in serializationschema
hotfix table common fix typos in docs of new source sink interfaces
new object create icon fix
update mybatisconfig java
hotfix runtime simplify slotpoolimpl mayberemaporphanedallocation
hotfix examples table fix indention in changelogcsvformat
use byteorder nativeorder for unsafe isbigendian0
hotfix table fix typos in tableenvironment javadoc
hotfix simplify the lambda expression in taskmanagerrunner
hotfix fix the access modifiers in taskmanagerrunner
feat merge upstream develop
fix a typo
improve mark all read performance
hotfix datastream api fix checkstyle issues and javadocs in checkpointlistener
reduce verbosity of job logs
fix theming issue with snackbar
fix a typo
remove unnecessary paragraph tag
better synchronisation and don t try to cancel uninitialized call targets
if available frames are less than requested don t try to fetch more frames
correct error in setting cu index for deopt targets
erd attribute figure spacing
add a namespace for scoping generic client metrics
add an isolate into the objectname as a separate key value pair
changed default for invalidateinstalledcode in ctw to true
substitutiontype should delegate to substituted type for hasdefaultmethods and declaresdefaultmethods
remove jsonobjectcodec s override
ensure experimental build server does not affect dry run
prevent attachment download button re animation
fix issue with some search results returning empty
fixup test case after merge painless extensions work differently now
fix create new pin option in registration flow
allow skipping if you hit a network error during pin restore
fix accidental checkin of debug code
refactor validate testcase
refactor validate testcase
implement closeable add testcase
condensed fields by remove extra space
add oldmethod for compatible
fix testcase compile error
update esproductdao java
修复talkback下tabview无法聚焦 talkback是google推出的android上的屏幕阅读器 透过talkback能让视力受损用户与android设备上的app交互 当tabview无法被talkback聚焦时 用户将无法与widget交互
rsv panel contents double refresh fix
修复talkback下聚焦到tabview时talkback不朗读tabview显示的文本 talkback是google推出的android上的屏幕阅读器 透过talkback能让视力受损用户与android设备上的app交互 当talkback聚焦tabview时 缺少contentdescription talkback将不会向用户朗读所聚焦的tabview显示的文本 用户将无法得知所聚焦tabview的含义
update test to be more specific
fix bug while customize webappstat
hotfix conf fix javadoc of checkpointconfig isunalignedcheckpointsenabled
fix postgresql partition ddl
bug fix for jsonobject tojavaobject
bug fix for jsonarray readobject
moved opening bracket up with class definition
moved jython cache directory from user temp to user settings directory
fix t parameter error in selectionsort
cleanup remove unused code
fix the documentation in polyglotnativeapi
fix checkstyle error
update isarray to first check if staticobject represents null
remove outdated exportmessage
handle case when type is null
adjust unsafe automatic substitutions canonicalization
work around broken image
replace explicit use of pooleddatabuffer release
using no alpha buffered image
fix exception when receiving poison_pill poison_pill is causing writing to closed client due to logic in code
fix unwindstack for non constant continuationstackpointer
remove unused import
reinitialize at runtime the ssl logging classes to support javax net debug configuration in generated native image
remove an unused import
use fallback for the fallback case in read writearraymember
remove comparison with zero from isnegativezero implementation
view toggle handler
the method ismemberinvocable string from the type eventcontextobject can be declared as static
fix the concurrency issue
injected accessors perform a null check
add class name to mbean error message
merge branch master of https gitee com linlinjava litemall
addressing code review comments
fix automatic unsafe substitutions
drop opt suffix from optimizedcalltarget tostring
update databasemessagesender java fix bug for log adjust the order of object creation to avoid creating useless objects
update defaultdiscoveryservice java add override
add economicmap putifabsent
fix compiler warnings
use encoded graph cache in partial evaluation
spawn different caches depending on the specified capacity
add simple test for economicmap putifabsent
schema selector fix use active editor s context
fix exclude directories from dex convertion results
make thread id a string
add documentation for hybrid
add hybrid layouts for other classes than dynamichub
fix rename test
remove httpclientrequest writequeuefull synchronisation when using stream
update memberattentionservice java
correct class name of stringtoperiodconvertertests
reformat comments comments are preferred to comments in this project to avoid strange re formats
do not log classinitalizerhassideeffectsexception
make compilerthreadidled protected
use timed wait to detect idleness only when core threads cannot timeout
fix auto full issue
set the name of the signal dispatcher thread in native image
update membercollectionservice java
muting failing apikeyintegtests
deleted excess line separator
fix graphql test
feat add test for assumefalse
fix consistent order of assumption and type getters
add documentation for identity hashcodes
highlight search suggestion
hotfix harden jardeletehandlertest test
invalidate tainted methods only when the encoded graph cache is enabled
bugfix only one qqface
bugfix default value
update membercollectionservice java
modify array size
remove beanbuilder loadbeans string this method is unused in the jenkins ecosystem it is potentially problematic and dangerous
renamed getbytes to getbytesfromstream
updated license year and javadoc format
clarify that the encodedgraphcachepurgedelay option has effect only on the hotspot truffle runtime
add since tag to the javadoc comment
avoid transient npes in unit tests
add missing but common vary header name signed off by paulo lopes pmlopes gmail com
used assertfalse in place of assertequals
add a method to modify the regular suites
don t run profileuploadjob if you re not registered
update omsportalorderreturnapplyservice java
avoid unnecessary property relocation when reapplying transitions
allow dynamic dynamicobject library dispatch via dynamicobjectimpl subclass
add identity comparison fast path to keyequals
do not include class name in generated assertion message
remove unused method
add note about hidden keys
no need to search the jvmciclassloader path for truffle services
polish code style
fix misspelling of references
avoid references comparison in isgreaterthanzero
trivial javadoc format correction
fix javac warnings
reset truffle compiler after running all encoded graph cache tests
force monitor slot for java lang object
handle empty group change byte array
sulong preserve source debug information for intrinsified functions
write barrier test disable partial unrolling
remove supress warnings
schedule verification do not consider dead floating nodes
safe point elimination always consider pre post loops for elimination
java remove debug line from cdp generator
update stringtobooleanunittest java
code cleanup unused import remove
make fillerobject immutable to avoid monitor field
fix returnlength assertion
update omspromotionservice java
initialize jdk classes before looking into the spec
add jdk vm to exceptions for the agent
initialize org ietf jgss at run time
remove initialization of jndi
fix reflection issues with java beans
no need to rerun in javax management
remove jmx from initialization list
initialize jfr as our substitutions need it
remove java sql
java beans introspector gets initialized through jmx
merge branch master of https gitee com elunez eladmin
remove aarch64arithmeticop frem
fix replace imports with single imports
update umsmemberservice java
allow createunwind on fast path
fix license header for datastreamrestit
remove blank tips of the day
add substitution to getbinding and setbinding
hotfix table planner blink fix digest for inline structured types with generics
add memory statistics output to the deadlock watchdog
register string getbytes for jni access at runtime
sulong fix long double and int float conversion in the tollvm node
in staticobject isstring just check the nullity and klass
prefix todisplaystring for interop objects with foreign object
throw if clone is called on an interop object
if a polyglot binding is not found return espresso null
throw in the non substituted versions of polyglot eval importobject exportobject
fix marshallingcodec throws illegalargumentexception river
update portalorderitemdao java
sulong llvminteropreadnode should perform little endian reads
jdwpcontextimpl getstringvalue should use todisplaystring
assert that an object s klass is null iff it s espresso or foreign null in staticobject isnull
restore accidentally deleted line
更新一下注释 更新一下注释 感觉之前写错了
update options java
add javadoc hint
fix merge error
handle null profile names better
update portalproductdao java
do not update threads that do not exist
fix minor typos ci skip
adjust array range write barrier checks
fix contact icon tint issues on older android versions
disable image heap card marking when using survivor spaces for now
fix use truffleweakreference to avoid boundary in reference get call
fix graph window placement issues
update espresso polyglot api documentation
allow exclamation marks as filter delimiters
remove leftover printstacktrace
initialized sun security mscapi at build time
fix regexsource hashcode
fix array s predicateiterable with collections allocateiterators
only show profile updates in active groups
only show delete action when long pressing on profile change update
use context for request timeout
check for description null
java fix logic of isinternetexplorer check
cleanup use shouldnotreachhere instead of new assertionerror
simplify new internalerror initcause e to new internalerror e
change excel file direcory
do not show typing indicators for inactive groups
remove star imports
polyglot cast null cannot be cast to a primitive type
polyglot cast fix the fields check throw if does not exist
polyglot cast null can be cast to a primitive type
fix a comment formatting
enable skin tone selection for emoji reactions
rework http client setimout
adapt to upstream changes
fix issue with gtreemodel s firenodechanged method not actually telling the tree which node changed
fix linuximageheapprovider image location failure with chunked image heap
enable chunked image heap layout
remove unused line
fix incorrect test
httptracertestbase should use now the correct types for client request response
make timeout fields private
remove no longer user sysprop names signed off by thomas segismont tsegismont gmail com
grid add prefixed file upload endpoint to node
fix dynamicobjectlibraryimpl keyequals
add substitution for the new signature of class getconstantpool
address checkstyle and findbugs warnings
update the branch to use java version capabilites check
update smscouponhistorydao java
updaate version in dep bank
remove clean up code the reached values should be gced after saturation is completed
fix for cppexporter running in headless
add canonicalizer during mid tier
disable rewriteduringcompilationtest testloopcompilation
fix createconnectionthread not exit issue
virtual primary key resolve fix
rwo refresh fix do not compare attribute props but name they may be different because of different select query
delete unused imports
sulong parsing fails now if data layout is not little endian
delete unused import
espresso polyglot api minor documentation improvements
add a convenience eval cast method to espresso polyglot api
add a convenience importobject cast method to espresso polyglot api
adapt to upstream changes
fix incorrect test
update druiddatasource java
fix proxy ref
add nonnull annotation to method argument co authored by tobias groza tobigr users noreply github com
allow tests with argument
fix profileoffset increment
data transfer column mapping respect database native case
fix npe in transfertointerpreter with engine tracetransfertointerpreter if called outside a call target
fix ambiguity in imageheapprovider determining image path when scanning proc self maps
writable constants fix
display error message if ld gold is missing remove unicode from source file
fix missing import
updated code to use only console
remove useless code
update copyright year of changed files
fix group change failure reason display logic
sync contactrecords as whitelisted if they re a system contact
sulong set version in llvmlanguage
improve writestream documentation
query execute monitoring messages
skip loadtimeweaverawareprocessor with native images
cdp less noise on the console when building things
rename shardingorchestrationlistener to orchestrationlistener
update springtaskconfig java
fix npe in statistics listener if compilation was never queued but directly compiled
javadoc add javadoc note explaining that stringutils cleanpath should not be depended on in security context
merge remote tracking branch origin devel into devel
add deletenode method
add another defaultpathvariable method
nat mapper logging add
ignore testloopcompilation test in case immediate compilation is enabled
for code format
remove getter for orchestrationfacade isoverwrite
graal test enable metric printing again
tests fix test failing due to recent refactoring
remove useless import
espresso polyglot api documentation update wording of polyglot availability
larger boundary for waitforreferencependinglist and explain the reasoning behind ignoring the interruptedexception
enable last tier compiles
ignore typing indicators from blocked group members
add system out
textfield don t update the display text when the text didn t change
deprecated method remove
oracle ddl extract improvement trim
row refresh fix cell extraction
update for zero and negative numbers
created seaprate method for each functionality
update shacommonutils java replace stringbuffer with stringbuilder and add initial capacity stringbuilder is faster because it is not synchronized
update globalcorsconfig java
updated with maths package
updated with maths package
cancel all monitors on platform shutdown
transaction management fix check txn support
transaction management fix check txn support
change test class name add blank line and swap assert method parameter
glyphlayout fix color markup not incrementing x
avoid use of static nativeimagesystemclassloader field
avoid using classloaderfeature singleton in lambdametafactoryrewritevisitor
explain disallowedclassloaders handling
remove redundant excludes filter use
update jacksonconfig java
oracle context init fix
progress monitor fix
progress monitor messages fix
progress monitor messages fix
merge remote tracking branch origin devel into devel
usagestatisticstest failure java vendor vs java vm vendor
disable card marking to unblock ci
fix bug causing profile updates to unarchive threads
fix issue where feature flag fetches weren t limited
reduce the number of cats in giphy sticker search results
forbid polyglot cast to abstract classes
changed variable names
do not init acra if inside its own process https github com acra acra wiki troubleshooting guide applicationoncreate
sulong fix unused try with resource warnings in tests
sulong listeners function fallthrough warning
sulong rawtypes in llvmruntimedebuginformation
sulong fix rawtypes warning in llvmcontext
disable starting player service via media button when there is nothing to play no active play queue
fix npe on group multi invite
filter bundles for source directories when asked for script dirs
don t compile script in event thread
script tests should wipe the directory after each test
use msg instead of stderr
fix for non unix paths
add more time to script tests
avoid multiple ghidrascriptutil initialize calls w multiple tools
check for existence of bundle paths on restorestate
bundle status tests fix path for windows
put ghidrascripttablemodel table changed events into the swing thread
set width of bundle status type column
disable undo redo buttons when no undos redos
fix after rebase
simplify hashcode example
print missing string stringtable debugstringindex
jniexceptionwrapper java lang nullpointerexception with jdbc
throw on an attempt to instantiate polyglot
update rabbitmqconfig java
make sure the user scripts dir exists before creating a script in it
hotfix table api java fix javadocs for tableenvironment fromvalues
polish code style
merge remote tracking branch origin devel into devel
remove invalid comment
made textfield getbackgrounddrawable protected
uncommenting the springboottest and test annotation
fail if lookupdeclaredmethod finds more than one method with the given name and arity
retrieving the object from optional object
fix typo in remappedrecords
pg filter partition indexes in table cache
determine if an object is of primitive type
determine if an object is of primitive type
determine if an object is of primitive type
update perfectbinarysearch java add package
refactor unit test name
fix correct parsing for array data payload
save preference on emoji send
properly render reset session update messages
fix different topics subscribed to the same redis node in cluster
query cancel force
update redisconfig java
fix no type check need for object
value transformation error handle
monitor log fix
search in object list fix re activate search action on folder switch
slightly better description of our alpha nan fuckery
corrected assertequals order for expected actual
update ignoreurlsconfig java
sync mockhttpservletresponse changes
javadocs on color about float colors using compressed alpha i assume no one ever reads numberutils javadocs
use public class hashmaplinearprobing
classvalue substitution fix computevalue usage
view toggle action fix do not activate views
mysql object stats sum data length and index length
remove java util objects calls
fix check redisconnection status befor redisconnection object join freeconnections signed off by xujie mikawudi qq com
merge branch master of github com redisson redisson
update copyright date
loop transformations limit unrolling to counted loops with a real counted loop exit
loop transformations better error msg
schedule verification do not consider dead floating nodes
make invokeespressonode s execute final to avoid overriding in the generated code
make toespressonode public
toespressonode do not coerce foreign nulls to espresso null
fix crash when opening video in local playlist tab
update resttemplateconfig java
fix copyright issues
typically command pattern is implemented using interfaces and concrete classes refactor the code to use the same
remove unwanted modifiers
hotfix mesos fix the access modifiers in launchablemesosworker
hotfix mesos remove the useless env variable in mesostaskexecutorrunner
fix checkstyle error
add serialversionuid for primarykeyposition
fix checkstyle warnings
update securityconfig java
fix processing of unsupported messages
removes usage of dictionary
clean up log statement in fcmfetchservice
bugfix current activity reference
fixes checkstyle error
implement instantiate for primitiveklass
update securityconfig java
simply call optimizedcalltarget call in the uncached optimizedindirectcallnode
simplify comments about encapsulatingnodereference usage in call
simplify the uncached defaultindirectcallnode since it is the same logic as defaultcalltarget call
remove unused constant in indirectcallnode
hana unique keys read fix
in instantiate check if the constructor is public
monitor logging fix
use collections emptyiterator in polyglotexceptionimpl
classvalue use better name for null replacement
use public class hashmap
hotfix table runtime blink return mutable instances for list converter
hotfix runtime fix typo in locationpreferenceslotselectionstrategy
ensure non intrinsified unsafe methods are inlined in checkindex test
grid cell focus rectangle fix
update copyright year of changed files
do not update thread on profile name change
adjust zoom levels and transition duration
hotfix remove unused futureutils runsync
add white space
hotfix docs rephrase the description about file names in streamingfilesink
remove nullable from cookielocaleresolver determinedefaultlocale by definition this method should not return null
fix queue might be null
merge remote tracking branch origin devel into devel
merge remote tracking branch origin devel into devel
add test case for memorymergedresulttest next
include a recipient in sent transcripts when possible
add generic for resultdecoratorengine
privatization catconstants java construction method
修复 httpstats 单例在并发环境下指令重排序问题 禁止 current new httpstats 指令重排序
polish javadoc for mergedannotations
sync mockserverhttpresponse changes
update reactivetypehandlertests due to deprecations in reactor
fix test case
some minor spelling error in comment
oracle table cache fix columns query
fix compile queue might be null if no call target was created yet
generic model fix generic types
stats render extra check
spreadsheet column dnd fix drag at any place
update hotel java
make method non static
add generic for routecontext sqlstatementcontext
improve uricomponents sanitizesource
allow overwriting plugins in pe decoding plugins
add clarifying comment
make the cds flag remote capable
move defaultsqlbuilder test case
fix word typo
active schema command refresh fix
disable erds for non relational schemas
property tree viewer edited value compare fix
don t fail cache lookup of a deleted node
add fix to update provider tab title on structure name edit
remove unused import
improve assert in looppartialunrolltest
replace pump occurence in javadoc by pipe
fix the sorting in java article
no repeat oops
hotfix python remove unused variables
another style fix
make log_of_heap_region_grain_bytes optional
preserve old behavior for old jdks
update cachenamespace java
hotfix table common enable hashcode equals for materialized raw value data
merge remote tracking branch origin devel into devel
force eager resolution for truffle compilations
allow optional codeobserver installation
remove timestamp when sharing a live stream
reduce cancellation detection frequency
update mainmenuunittest java updated the method name to follow bdd conventions
hotfix remove mocking from restserverendpointitcase
avoid usage of accessor methods in dynamicobjectbasic
fix call parse instead of valueof for option parsing
disable compiler idle delay if statistics collection is enabled
hotfix test fix formatting in checkpointrequestdecidertest
fix accidently remove checkhotspotcalls
update miniocontroller java
allows bootsrapmethodattribute to specify a dynamic constant as a static argument
statistics render fix use plain font
fix unsupported exception in decompiler provider
sql server statistics
sql server statistics
hotfix runtime guard resourcemanager startservicesonleadership from being overridden
hotfix runtime make resourcemanager onstart final
bugfix should refresh thumbview when settickcount
replaced string with charsequence
bugfix click event is prevented if movementmethod is not set
boundary for jvm_getnanotimeadjustment
pg dump case sensitive schemas backup fix
update group size flag
polish method order
create enter and create exit is enought as the receivers are immutable
add close discover switch
regression close player in onplaybackshutdown
fix an issue with the order of imports that was causing build failures
update and rename heap_sort java to heapsort java
fix array initialization
fix array initialization
older eclipse version support fix
stats renderer use explicit grey colors for host name macos gtk
fix a whitespace and spelling issue that was causing the test case to fail
improved an animation logic
update application java minor javadoc fixings
hotfix fix typo
update cancelorderreceiver java
update cancelordersender java
fix javadoc formatting in genericgroovyxmlcontextloader
nests must have same runtime package loading constraints for an interface method checks also for constraints on the original interface
do not push monitor of synchronized method will be freed in finally prevents double exits
private methods do not override
fix npe when double tapping the conversation pinning icon
prevent swipe actions on the pinned header
document based resultset model fix
attribute binding fix
number transformer fix use pojo
update smurfsarchunittest java
remove useless code
partial evaluation assert graph is schedulable after pe
merge explode proxy test refactorings
java preventing modification of values supposed to be constants
java deleting unused imports
remove invalid assertion
extract common part from if
corrects meta resolveandaccesschecks did not handle arrays
cdp throw an exception if unable to process a message
remove obsolete assert
remove useless suppresswarnings for deprecation in verticle
remove bundlehost listener on plugin disposal
update copyright year of changed files
update copyright year of changed files
hotfix yarn remove incorrect javadoc
add back remove tests
invokedynamiccallsite check if the result is a foreign object
add truffle boundary to avoid runnable run being included in compilation analysis
add truffle boundary
merge remote tracking branch origin devel into devel
update classloaderwrapper java
minor formatting fixes
monitor logging fix
fix a test case issue within the dirty flag module
update arraytolistconversion java
for aot honor aotverifyoops option and ignore verifyoops flag
update classloaderwrapper java
mark threadspawnrootnode internal
minor formatting mistake
a comment on field access modification for espresso objects
add explain context
espresso polyglot api update the explanation of casting to an array type
document casting foreign values to arrays in polyglot
use single signed off by johno crawford johno crawford gmail com
allow subclasses of hosttoguestrootnode to control when exceptions should be wrapped in guesttohostexception
refactor shardingspherealgorithmfactorybean constructor s sequence
use more parenthesis to be safe
results viewer zoom actions
remove useless todo
javadoc for reportpolymorphism megamorphic
explicitly do not use jna
explicitly build terminal with jansi
don t use system terminal
full name in onlywith
substitute universaldetector as well
add better licensing info
update year printed in polyglot shell
avoid npe when printing errors in polyglot shell
recover sqlparserparameterizedtest checktestcases
write mode for modifying set signed off by johno crawford johno crawford gmail com
typo fix skip ci
fix occurrences of typo
fix jni virtual method call comment on jvm_getnanotimeadjustment
improve espresso polyglot api documentation
fix small gtree call that needed to be in swing thread
atomic sql operations fix
atomic sql operations fix
bandaid fix for calltreepovider stack trace
merge remote tracking branch origin devel into devel
test signed off by abhishek chandrasekaran abchandrasekar expediagroup com
change ut signed off by abhishek chandrasekaran abchandrasekar expediagroup com
use requiredargsconstructor for schemacontext
add parse columnnameswithsort method
tinker clean unused imports
rename property name
single entity recognition fix
fix cannot clear threads on close contexts mights still need to leave
merge branch master of https github com psevestre tutorials
remove unit test errors
fix check style error check
generic connection page fix npe
fix trufflefeature was referring to old class name
fix textarea prefheight sometimes resulting in wrong number of visible lines
cleanup context and context thread locals on close
fix comment in truffleinstrument
revert visibility change to doclookup constructor this can remain package private
fix spawn node to insert call node in execute method update test assertion accordingly
rotate the cds feature flag
fix crash when archiving multiple conversations
add null check is no longer needed after previous commit
change return type signed off by abhishek chandrasekaran abchandrasekar expediagroup com
fix create statement for recipientdatabase
mark randomnumbersfeature as an automaticfeature
change as review
remove blank line
remove blank line
changes after review
fix reading a boolean from a foreign boolean array
rename methods in metricsutilstest
make cache invalidation test more robust
fix action bar usability in vertical screen split
update maths armstrong java
fix crash when scrolling to the top of a conversation
auth model selector ui fix
use standard location for errno h inclusion
workaround false gcc compiler warning
make older versions of cl exe accept bitfields larger than int size
eliminate need for struct forward declarations
suppress warnings when querying bool types on windows
tune strict compiler options for windows
fix appendix typos in contentdisposition
add a note about weak references to threads
fix api javadoc
improvement get already existing sortingtrackgroups if exists
update pmsproductattributeparam java
fix null scriptinfo when a key is assigned to a script
fix wrong assertion
fix a test
make group links remote capable
sql error logging improvement error chain
rename statuscontainedrule to rule
fix init factor for map
move authenticationchangedevent to auth package
update java maths primefactorization java
update pmsproductcategoryparam java
remove debug println from build logic
call reportloopcount also in first tier for optimizedloopnode
add unit test to prevent shipping forced feature flags
fix throw illegal state instead of assertion on entered assertion
fix context tostring format
make context local test more robust
be careful of recursive entry
equals should be called with a constant
add to username in preferences
fix pg的select语句带orderby时候 解析出来的sqlorderby的parent都是null
fix pg的select语句带orderby时候 解析出来的sqlorderby的parent都是null
ensure stamp of compressionnode only gets stronger
remove unused fixedvalueanchornode
polish code style
fix add test case about defaultobjectfactorytest
fix polyglotthreadinfo enteredcount is read from multiple threads and needs to be volatile
add logprefix to the log statement add logprefix to the log statement to provide better debugging experience
reset select info
improve setcachedthreadinfo should always check for cached
make cleanup of contexts more robust add comment about isactive
fix for string equals check
invalidate the noforeignobjects assumption if the receiver is foreign
wasm wasmexit constructor needs a truffleboundary
wasm replace sun reflect generics reflectiveobjects notimplementedexception with wasmexecutionexception
fix class cast exception when member is approved
wasm mark binaryparserexception as never part of compilation put truffleboundary on assert fail
sulong conflict for register read only globals
sulong return the file scope of the library if the library has already been loaded
sulong put truffle boundary for blacklisted svm method
sulong ensure libraries are still initialised even when the call target of the load module node is called with arguments
sulong created an exception when an internal library is add to the context as an external library
sulong minor formatting
sulong minor formatting
sulong add more debugging info for unsupported messages also add more synchronizes for add native libraries to nfi context
sulong synchronizing over the collection of native libraries when they nfi is attemping to add a new library
sulong minor formatting
sulong minor javadoc
sulong remove the extra exception being thrown for unsupported messages
sulong testing the gate to produce more information regarding an unsupported message exception
sulong tmp commenting out unused method
sulong update copyright and set truffle boundary for black listed methods
sulong add more truffle boundaries to allow for stack trace for unsupported message exception
sulong remove unused argument
hotfix network fix field visibility
bugfix use textcolor
reset select info
update umsadminloginparam java
splitted apiinfo to multiple lines
redundant context commit remove
add close discover switch
update arraystoreexceptionexample java
blank line add above member definition
fix index masking
remove unuse mock
add support for detecting icc
modify to prevent memory leaks
process reviewer comments
update umsadminparam java
update blogcontroller java update formatting
disabling tomcat url stream factory
fix typo in comment
change assert in replaceoverride constructor from methodname to methodreplacerbeanname
explain the behaviour of casting to wrapper classes in the polyglot documentation
clarify varargs in proxyarray fromarray
propagate snippet loop frame state assignment to loop exits co authored by david leopoldseder david leopoldseder oracle com
fix optimizedassumption awaitdependency to not trigger an interrupt loop
ensure all types are linked
fix pg dcl
remove unnecessary word from text
further explain the idea of fatalcontext in javadoc
sort nodes according to id if the frequency is the same
add braces to the if block
don t throw an error if the same method is prepared for jit compilation multiple times
progress logging fix
read only status check
fix it some changes
fix fatalcontext default behaviour
remove unuse import
prevent jumping of the player on devices with cutout
update updateadminpasswordparam java
hotfix task add suppresswarnings to streammultipleinputprocessor
eagerly convert a foreign value to string when polyglot cast ing to j l string
explain eager conversion of foreign strings
throw shouldnotreachhere on a cast to void because a cast to void cannot originate from the user s code
replace with constant replace allow range query with inline sharding with constant
ignore link preview descriptions that match the title
ensure feature flag is string before cast
fix issue for data source cannot renew
fix test case
reassociation phase can increase code size if shift add sub to mul rewrite occurs
synchronized deadlock fix
grid testing skip checks for content type works skip ci
progress monitor log fix
add imagesymbolsprefix hosted option
data load progress visualization fix
grid rendering fixes lines
grid add router role to hub this enables the flags on the cli
fix accidental change to interpolation pow3ininverse
devirtualizing interface call should respect the mayspeculate flag
update osscontroller java
warn message add
forgot to run the app
update dynamicprogramming minimumsumpartition java co authored by du yuanchao shellhub me gmail com
pinned column selection fix
test compile fix
test compile fix
fencing ignore fix access if it has a guard
fencing run test only on supported platforms
use mockeddatasource in proxyschemacontextstest
add abstractnewobjectnode setfillcontents
only run a test if the docker server is running
remove singleton for shardingsphereproxy
update commonpage java
reduce method size to favor inlining
auth model obtain fix
update sorts bucketsort java
deprecated methods remove
json config is deprecated
refactor mysqlcomquerypacketexecutor mysqlcomstmtexecuteexecutor
refactor proxyschemacontexts s constructor
java have grid console permanently redirect to the new ui url
temporary fix for running jck
add a note about the deadlock risk
boundaries on exits
add math signum intrinsics check to checkgraalintrinsics java
context finalization still calls into vm teardown
hotfix remove unused import in checkpointcoordinatortestingutils
hotfix make it more clear that the master hooks are also fired in the checkpoint timer thread
reduce apngparser logging
fix code style
fix check style
support pg create table not exists
hotfix make logical slot dummypayload public to reuse in other tests
refine prevalidate log
fixes unit test for mysql insert statement
update commonresult java
fixes unit test for mysql insert statement
fixes unit test for mysql insert statement
dt mappings fix npe
update test to use loweredatomicreadandaddnode
remove unused imports
fix pip positioning in video calls
suppress exception in finalizecontext
hotfix runtime add missing nullable annotations to vertex colocation group
remove unuse import
test fix for recent test framework change allow re entrant swing thread calls
ignore close on null dialog
fix wrong timestamp being used when trimming by length
use proper lifecycle for simpletask which touches fragment view
fix unarchive menu action
corrected processor manual url generation
prevent seeking out of video duration in player
assert upon inlining that callee and caller share the same speculation log
disable mentions megaphone
delete unused import
remove unused imports
test avoid double null check in transportsearchactiontests
remove blank space
disable a test that does not work in chrome for edgium
update after rsocket snapshot changes
fix bug for proxy hung
hotfix javadocs improve javadocs for streamexecutionenvironment addsource
ignore framestate when matching nodes
corrected handling of stored null pointer within processor test control blocks
on connection close exceptions make sure thread is interrupted to tear down gracefully
hotfix javadocs remove inexistent memory unit reference from memorysize java there is no q unit on memoryunit enum
hotfix fix checkstyle formatting
hotfix fix checkstyle violations
corrected emulation tests for thumb code
handle error case for invalid function pointer
remove useless log
reassociation left shift node document tryreplacewithmulnode
grid only pass available hosts to the host selector
grid wrap reading the model in a read lock
grid make the host selector a field
delete unused code in dataclassrowmapper
suppress deprecation warning in abstractdatabufferallocatingtests
muting coordinatortests testlogsmessagesifpublicationdelayed
sulong profile llvmpointerimpl dispatch on native pointers
sulong align stack allocations from lukas stadler
sulong nativetypelibrary is being used here
sulong refactor isinstance in llvmmanagedpointer and llvmnativepointer
sulong minor changes
sulong add a create method to llvmrealloc
changed save project action to have the same save behavior as closing the front end
introduce analysistype isreachable method
assertisnotserialexecutewhenintransactionandbasetransactiontype assertisserialexecutewhenintransactionandlocaltransactiontype and assertisserialexecutewhenintransactionandxatransactiontype add
hotfix connectors improve javadocs for singlethreadfetchermanager
muting logstashsystemindexit testpipelinecrud
load indexed node refactorings
hotfix consider initializing jobstate also as scheduled in clifrontend
ignore a test that is hard to prove is safe to run
drop the use of frozen type states in nodeclassfeature
grid add tostring to nodeid and slotid
fix for untouchable area
typo fix in logged bundlehost error
add license header
update twopointersalgo java
add safepoint id
truffle add readonlybufferexception to skipped exceptions fixes pe of the isreadonly check in bytebuffer varhandles
put stringbuilder append behind truffleboundary
bug fix for scheduler keepalive true
update swaggerconfig java
fix logging call placeholders number
json increase max depth of recursion for json output
performance improve using bulk operation instead of iteration if possible terminate loop after condition is met
code cleanup remove unnessary boxing
enable disable video after screen on off regardless of background playback setting
ensure no memory reads are scheduled before the c to java transition
update passwordgen java
remove unnecessary suppresswarnings
grid avoid hard coding a port in a test
when sending json omit class names
consistent flushing of given outputstream
document persistence support
add a new section in lombok builder article minor fixes
navigator tree update fix
double page activate fix
fix compilation issue in versionstringfieldmapper
s fomrat format
use non patched method bytecode for methods being redefined due to override methods being add remove
use jdbcutils isoracledbtype
hotfix runtime remove commented out annotation in sourceoperator
get element at special index
use empty string for name if can t obtain
remove unused imports
remove unused import
inner class messageeventfactory to be static
sql editor action buttons enablement fix
add a comment signed off by paulo lopes pmlopes gmail com
get vtable from redefinition cache when verifying
update esproductservice java
hotfix datastream chaining serialization exception during job graph generation
hotfix network annotate networksequenceviewreader getnextbuffer as nullable
mute clusterhealthit testhealthonmasterfailover while we await a fix
checkstyle error fix
mute appendprocessortests testappendingtolistwithduplicatesdisallowed
don t show the link preview megaphone to new users
prevent npe when setting sticker emoji
fix npe in ghidraapplicationlayout when the layout is used from the ghidradev eclipse plugin
fix crash in retrieveprofilejob
ensure that declaring class of all unsafe accessed fields is reachable
register constant s type as reachable
update maxsizeconstraintvalidator java
update maxsizeconstraintvalidator java
avoid assertj deprecation warning
leave args file in current folder and print its location
make host interop ismetainstance accept null argument
double check erroneous initialization under initialization lock
add shortcut for instanceof checks with final classes
remove cached host utc time zone
fix existing unit tests
share group link via signal to one recipient
remove unnecessary silent throw
fix leave group crash
fix false group name and avatar updates
fix stale call preview state by finishing when leaving
update esproductserviceimpl java
fixes with clause error message
fixes code conflict
tinker fix classloader in base context was not replaced
disabled nested scrolling of queue
hide type for null objects without meta object
update license header
add unit test for configcenter
merge loop for target mapping class cache initialization
fix copyright headers
fix media preview crash
fix bug in tests that allowed for concurrentmodificationexception
treat invokecfunctionpointer the same as cfunction for stack overflow checks
update http status codes expected in failover scenarios
prevent multiple module root entries when launched from debugger
ignore services without a nullary constructor
defend against typenotpresentexception in reflection data builder
provide a setter for threadlocalrandomaccessors and splittablerandom
clean failure doc url link
migrate the deprecated method to the new method
fix auto queue with no next videos
replace all failure doc url link
grid only check hostname for connections that use host names
grid the oneshotnode now indicates it is draining on starting a session
grid rejig the internals of localdistributor
grid replace mistaken wording no logical changes
hostexecutenode fix argument count validation in the uncached specialization
update esproductdao java
ensure clock adjustments does not stop remote config refresh
do not force inline call to function in foreign call node
add missing copyright notices
grid listen out for a node being remove in the model
fix missing submit debug log loading progress spinner
change aborted to failed under testfailed
explain plan from query and statistics render fix
fix crosslot error thrown if rliveobject got updated
default optimizedcalltarget isalive to optimizedcalltarget isvalid
ignore bad messages from blocked senders
remove unnecessary stubbing
fix spinner not disappearing when add members with no network
option parsing bugfix
allow in notification reply to multi message if you can reply to latest
parse validate the name of a custom section
validationsuite should be executed within wasmtestsuite
add tests for the validation of the name of a custom section
replaced two more urls updated license headers remove extra blank lines
clear auth cache on first verification failure
reply with correct error code when this was garbage collected when trying to invoke an instance method
hotfix minor streamgraphgenerator refactoring
fix javadoc in internaldistributionbwcsetupplugin
sulong style format
sulong native files can have a null path for the bitcode parser the nfi will find the path
sulong command line libraries can be native libraries
add comment above parameterised constructor
minor issue fix
fix another issue
report loop count also for first tier
substitute executable parseparameterannotations
remove outdated substitution for time zones
less locking signed off by paulo lopes pmlopes gmail com
project variables resolve fix
add the cause to the ise signed off by paulo lopes pmlopes gmail com
initialize player notification asap otherwise android s foreground services implementation would complain
fix spelling error
hotfix datastream api remove ineffective and unneeded serializability from streamnode the streamnode implements serializable but contains non serializable structures
set log level to fine to not provoke futher errors
update license header
update esproduct java
unused message remove
sulong remove caching on truffle files maybe need it for code sharing later
remove left over debugging code
fix llvm float convert
make creating assumptions for remove methods lazy
unschedulable graph test avoid optimistic opts to run correctly in the presence of xcomp
introduce local var instead of two calls to same method
javadoc for substrateoptimizedcalltarget isalive
add ordering for sybase stored procedure text to fix the procedure source not being displayed correctly
fix class desiredassertionstatus intrinsic
fix crash on fast forward
hide player notification when time it is useless to see how much time ago a player notification was created
block unknown group messages from blocked senders
help location fix
remove never used stateafter field in substratearraycopyofwithexceptionnode
fix arrayindedoutofbounds bug
pass the classloader and not null in nest members check
ensure we refresh recipients after changing storage keys
ignore innerclasses attribute check mimicking hotspot
bug fix for sql wall
wall filter add sqlite support
improved kylin support
fix duplicate scope id error with undefined functions in decompiler
change log level message
moduleaccess add comments
minor fix checkstyle in stateutil
pg default parameters trim
add stack trace to exception to get more diagnostic output
null check the receiver in arraylengthnode
make textfieldmapper fast_phrase_suffix private
schedule another attribute refresh for gv2
sulong llvmpointerimpl should be final
sulong pe couldn t inline the virtual calls to llvmasforeignlibrary isforeign
add deprecate method for compatible
refactor for performance
fix checkstyle warnings
add missing alias of llvm frameaddress builtin
restore eager creation on remove by redefinition assumptions on methods
method canoverride also checks for self non private non static
mute diskthresholddeciderit testhighwatermarknotexceeded
add unit to the class name to satisfy the pmd xml violation
extract uncached dynamicobjectlibrary to a constant
allow remote deletes of pending messages
remove compilationfinal after refactoring to make the field real final
fix assert for klass invokemember
register graalcompilerdirectives infirsttier intrinsic on svm
link a jbs issue for referencing the hotspot differentiation for mh resolving
fix unwanted change
fix debug context header
disable mention clicks in multi select mode
check index of indys in bootstrap method attribute array
remove unused method after refactoring
lps algorithm fix
finish cleanup of now unused getopcode in quicknode
use replace instead of insert when replacing quick nodes after redefinition
moduleaccess minor refactoring
ensure com sun jndi dns dnsclient is initialized at runtime as it contains a securerandom instance
ask for confirmation before clearing queue even when there is only one video in it
add verification for customjavaargs
increase width of hashcode field so that columns stay aligned
don t put spaces around property separator
shorten directcallnodes to calls
shorten codeaddress to addr
shorten tier names
hotfix tests return empty array of gates instead of null
hotfix tests report failurereason in streamtask tests
fix of recent change to window manager
add two finger to close player gesture
update esproductattributevalue java
two finger gesture less code lines
start time feature
add a test case checking the refusal of a malformed encoding of an export name
more specific error messages for gwt failure
update the license header
move remove listener logic to servicemanager
hide bytecodeswitch class not meant to be used directly
fix dup bytecode it should work with both objects and return addresses
hotfix mesos replace deprecated untypedactor with untypedabstractactor for mesosresourcemanager akkaadapter
hotfix fix the generic parameter declaration of parquetavrostreamingfilesinkitcase
hotfix rename parquetstreamingfilesinkitcase
remove debugging logic
vertxhandler should close using channelhandlercontext instead the channel when it closes upon an exception caught
for a remove method return the latest known method version when callers ask for info regarding the obsolete method
fix flaky arraylistmultimaptest
register correct providers for isolated compilation
correct deleting of buffer until end of newline
minor minor fixes in javadocs and an exception message in sortingdatainput
improve documentation for polyglotexception isinterrupted
fix description in tryblocktag
add comment to slparseerror about why interoplibrary isexception is not exported
improve documentation for causes passed to abstracttruffleexception
merge remote tracking branch origin devel into devel
hotfix checkpoint add testlogger to all checkpoint tests
fix jck vm verifier build issues
hotfix core annotate managedmemoryusecase as internal
sulong minor don t add call target dependencies that already exists
sort queue by call and loop count
include only non canceled jobs in queue size
fix race condition in type reachability registration
document unused parameters
remove class notation for class names
mark unused methods
apply api change due to rebase
remove replaces attribute for specialization
add error message if declared method has no implementation
fix checkstyle errors
remove string concatenation
first tier priority queue on by default
refactor type reachability rules
only tear down isolate if isolate compilation is enabled
update comment for substratetrufflecompiler teardown
make usecardrememberedsetheap part of the apioption group substrateoptions gcgroup
fix apioptiongroup single default error message
throw correct exception on invoke message of non class pointers
dynamicobjectlibrary do not report key cache polymorphism
hotfix test do not hide original exception in shufflecompressionitcase
corrected elf npe
fix bug causing incorrect mention suggestions
fix typo in pdbuniversalanalyzer java platform indepent platform independent
export wasmtrap as an error
fix style issues
add videoallcallback oncomplete
increment call count for call nodes in first tier compilations
update esproductrelatedinfo java
hotfix drop unnecessary powermockrunner from rmqsourcetest
fix test with dates low numbers of milliseconds since epoch were being interpretted as a year
feature streammessageid auto_generated const add
sulong disable generated bitcode tests that depend on ub
use gc value for gcgroup
reword description of serial gc
more correct coverage in compiled and corner case
add missing javadoc
simplify code for simpleclienthttprequestfactory prepareconnection
remove no longer needed method
fix static method warning in benchmark
fix only print warning for explicit context preinitialization
remove unused method
remove useless comment
fix invalid reference in javadoc
fix typo in inline comment
add comment explaining why we use a weak reference for the polyglot source
remove unused accessor methods
update storeengine flag on patch to use weak references if no storage is needed
update copyright headers
use assertequals in tests if arguments might be primitives
force invalidate to update call node count to be more precise
fix invalid space
fix single context function literals never updated the cached function
fix missing since tags on equals and hashcode
hotfix disable kryoserializerconcurrencycheckinactiveitcase if logging level is debug
update agp version to avoid warning also gdx ai
fix possible crash for unregistered devices
remove unused code
dynamic type still needs to be objecttype for the legacy layout
fix wrong delay check
reintroduce cancelexecution getlocation as it is accessed by reflection
remove unused import
update esproductrepository java
suppress ci errors for unchecked cast from object to map string object
reintroduce suppresswarnings unchecked for parsetoplevelobject
print stacktrace of possibly swallowed exception in contextinterruptparameterizedtest
update maxsizeconstraintvalidator java
disable graal specific plugins in libgraal
allow binarygraphprinter to call snippetresolvedjavamethod getcode
bugfix use wrong resource
fix illegalstateexception after onsaveinstancestate
don t rearrange lists by dragging the thumbnails
reflection usage fix
debug log fix in file
translate the numeric notation
fix off by one error in table
remove redundant instantiatestreaming and compilestreaming methods
license header add
add a comment to explain why osr loop call targets are not allowed to trigger compilation
fix the eventfiringwebdriver and therefore the failing test
fix invalidate also needs to cancel compilations if the compilation isactive
improving the context switching related comment
fix compilation error
drop debugging output
fix glframebufferbuilder visibility
update basename for messagesource
calling registernativemethods in the libgraalscope
swipe checks in divisions
disable multi tier if compile immediatly is enabled
call into last tier compile when compilation is synchronous
document that allowing implementations implies allowing access
update value as class javadoc
remove unnecessary overload
hotfix test remove unused code in alternatingcheckpointbarrierhandlertest
grid better json encoding of event data
do not enable admin approval on group links by default
hotfix tests improve condition readability
merge branch develop of github com alibaba arouter into develop develop of github com alibaba arouter fix postcard addflags bug
add docs on why executorservicewrapper is needed
disable unit tests for the encoded graph cache
grid only start the standalone server once
conditional elimination add docu
fix spotbugs detected issue
partial escape analysis add documentation
fixes after rebase
correctly fail if libeden so is not loaded properly
logging monitor fix
trim the adminaddress before saving signed off by darin pope darin planetpope com
avoid arbitrary implicit exception on sparc
fix npe in instrumentationtestlanguage initializethread
add runtime option activeprocessorcount to override processor count
hotfix runtime docs fix javadoc
hotfix runtime add missing test checking the proper implementation of unsafememorybudget getavailablememorysize
column type change fix npe
weaken the assertion in source equals for now
fix trufflecontext enter leave deprecations
only run start function when reinitializing memory
remove redundant requirenonnull
hotfix table common provide a consumed data type for supportswritingmetadata
node dispose utility
ignore some more zkgroup dependent tests on mac
remove remote delete option for group updates
reduce verbosity of keyboardawarelinearlayout logs
improved ads support
fix checkstyle violations
ensure no installedcodeobserver factory can be add at runtime
referenceinternals provide field offsets directly
add clickhouse mergesql testcase
sulong allow allowexperimentaloptions for test suite harness
sulong copyright for nfi
sulong update copyright date
sulong add allowexperimentaloptions for managed testing suite
allow disabling libgraalhotspotgraalmanagement completely
enable graph cache
deleted unused field in defaultlistablebeanfactorytests
make substratesegfaulthandler also act on sigbus on posix
correct outdated error message
do not check for blank
ensure that the help message is printed
test build error
complete build action
change default tracing policy for server to always
improve code quality
sql editor row count calc in doc dbs
fix column resize
fix colour names
use import for permission
delete trailing whitespace
attempt to fix failing ci builds
pageutils support hive odps
bug mysql解析引擎对while end while后丢失 导致执行时报错
bug oracle解析引擎对while end后丢失 导致执行时报错
attempt to fix flaky logging tests
merge remote tracking branch origin devel into devel
ignore libgraal threads in enginesgcedtest
reactive tests fix
workaround fix reference bug
minor add javadocs to cep patternstream
grid remove unused imports skip ci
add null check
add after review
do not show modern profile sharing on brand new conversations
preventing sending cluster nodes to the same host signed off by sergey peretyatko sergey peretyatko com
support jvm mode when polybench is a native image
format with google formatter
add configurationproperty tostring for debugging update configurationproperty with a sensible tostring to help debug task failures
bugfix fix mybatis param import error fix integer object compare use error
fix gui clear jumps history on file close
grid remove unused imports skip ci
grid fix test by add registration secret
hotifx runtime print root cause of logical slot allocation failures
hotfix tests let slotsharingexecutionslotallocatortest extend testlogger
fix remove unused field
rethrow interrupted exception in pause action
grid the data structure is immutable and does not need to be guarded for threads
expose webassembly object in polyglot bindings
add a missing truffleboundary
rename getentrypoint to inferentrypoint
mitigate pstn callback crash when service is in background
fix security crash during directory refresh
add missing allowallaccess to the polybench context builder
remove jdk class standardcharsets from classpath
allow all languages
add more reasonable defaults to polybench
include additional system properties in debuglog
have websocketimplbase closed field private as the subclass does not need it
remove debug logging in globalbuildinfo plugin
fix the end point of span exceeds the length of spannablestringbuilder
hotfix checkpointing remove unused variable in alternatingcontroller
add more huawei phones to the camerax blacklist
specify a special call for isfinalflagset
mark all methods of a remove inner class as remove by redefinition to avoid calling into obsolete code
serviceloaderfeature exclude service provider com sun jndi rmi registry registrycontextfactory
unmute already fix rartermsaggregatortests
do not use remove option
remove unneeded annotation
fix missing import
add missing override method
validating the correct order of sections
add regression tests for the correct order of sections
add missing throw to fix spot bugs errors
fix npe in backuputil
default tracing policy for http server is always
task last run format fix npe
grid add docker host flag to help command skip ci
grid add video docker image and path flags for docker skip ci
add comment about temporary assert and current workaround
update cmssubjectcontroller java
update versions in bank and depbank
add error info throw exception
replace a contextcompat with layoutinflater from
simplify an if else
bugfix draw indicator first
fix typo in conditions
grid fix npe skip ci
native image slow path alloc must kill tlab
nativeimageagentjnihandleset use save global object handle to java lang class
reduce parametrized runs for byte array benchmarks
fix copyright headers
update miniocontroller java
disable name checks in the region
clarify why we use two different system properties signed off by jorge vicente cantero jorgevc fastmail es
wasm only reset operand stack slots in partially evaluated code
bug 修复oracle的elsif 输出成else if
hotfix table common add changelogmode tostring
switching default http client to reactor
checkstyle optimize imports
sulong remove exception in language for disposing context as some contexts does not need to be cleaned up when disposing
check on symbol and not string for init
hotfix coordination remove confusing bulk from test names in physicalslotproviderimpltest
fix failing test
hotfix minor javadocs fixes for file source classes
register classes exposed via getpublicclasses and getdeclaredclasses calls in reflectionregistryadapter
create a duplicate methodversion for split methods
kill tlab locations in every safepoint slowpath
memory anchors must not be remove after the fixreadsphase
update omsorderreturnapplycontroller java
fix grouping of javadoc for org graalvm nativeimage hosted
s native image native image to keep consistent
grid improving imports skip ci
put full demangle in fid comment
update theme settings on data refresh
fix don t apply const split if not needed
clear resourcebundle cachelist field in the final image
hotfix only close the sourcereader and eventtimelogic in the sourceoperator if they are not null
fix crash around creating mms groups
java fix type mismatch in an assertion in cdp tests
prevent silent addressoutofboundsexception in mipsaddressanalyzer
java disabling a couple of tests in chrome that look like regression bugs
remove unused methods from queryshardcontext
hotfix core instantiationutil serializeobject creates objectoutputstream only when necessary
hotfix tests chain unexpected exception to assertion error
hotfix util fix precondition checknotnull annotations
hotfix tests remove unused methods in channelpersistenceitcase
remove unused field
hotfix coordination skip logging of excess resources if none exist
add changes to remove unwanted spaces
improved oceanbase_oracle support
hotfix in filebucketwriter change the bucket part id to uuid this provides more uniqueness than just randomstringutils
hotfix improve javadocs of sourcereader
introduce trap in deadendop
sql constants unify
cache library parses
hotfix table common improve terminology for data types in formats
java ignoring exceptions in shutdown hook
minor rework javadoc in sink interface
hotfix table common add logicaltypechecks haswelldefinedstring
hotfix table common improve supportspartitioning documentation remove confusing prefix key term from the class javadoc
update omsordersettingcontroller java
fix brightness and volume scroll swapped
fix tapping on video thumbnail does nothing
hotfix fix streamingexecutionfilesinkitcase instability
add tests for the validation of the correct number of code entries
hotfix add mocksplitenumeratorcontext runincoordinatorthread implementation this commit adds a missing method implementation for splitenumeratorcontext runincoordinatorthread
monitor logging fix
remove unused logger objects
java deleting unused imports
fix a memory leak in shared context mode
add logging statement for renaming of inner classes
change loop index from float to int
hide read more when long message is remote deleted
use a static import for better readability
fix potential npe
fix plugininfo bwc
fix duplicate semicolon
fix diagnostics agent error when both trace class initialization and trace object instantaition are used
ensure java version of graalvm is shown when driver runs as image
rename reptile to amphibian
fix issue where we were double logging job info
rename reptile to amphibian
java deleting unused private fields
hotfix connector files remove empty default method override in filesourcereader
hotfix close file formatreaders also when the source reader is closed before end of split
strict nullability for field assignment
fix checkstyle violation
sulong change equalsignorecase to equals for start and context initialize and dispose function look up
oracle alter table rename to miss space
more tests and cleanup
move isheadless call to the invocation plugin s apply method
use native image builder awt system properties
update osscontroller java
options that are more than one letter have two dashes
simplify abstractaspectjadvice isvariablename
grid retrying session creation for all cases
java deleting unused imports
java allowing a subclass to use its own logger
java abstract class does not need a public constructor
java fix double check lock in a lazy initializer
use methodfilter and in transactionaltestexecutionlistener
delete old logic
add equals method
minor fix the test name of sourcereaderbasetest testpollnextreturnmoreavailablewhenallsplitfetcherclosewithleftoverelementinqueue
polish javadoc for injectionmetadata forelements
assert same instance returned for cached merged beandefinition
fix race on inserting array store node
override isfinalflagset in klass to circumvent cha limitation
make the array store node field volatile
remove the intermittentests category from these no longer intermittent tests
fix long comparator in manyifsprogram
add truffle configuration service loading during image build
fix class name on initialization error
grid remove unused imports skip ci
update pmsbrandcontroller java
hex editor ioobe fix
fix racy test
monitor logging fix
fix metrics racy tests
jdbcutils close connection skip sqlrecoverableexception
update pmsproductattributecategorycontroller java
fix racy test
regression fix videos add multiple times to detail fragment stack
formatted with google java formatter
remove compilationfinal annotation on the table elements
change log level to debug
allow construction of derived induction variables from narrows
remove add context from engine if context locals initialization fails
move contextaddedtoengine variable declaration before try block fix merge problem
simplify astutils getpropertyaccessorstotry
protection domain check thread safe
java following java naming convention
java deleting unused imports
correct the package name
improve import package names
java fix code example in chromedriver javadoc skip ci
neverpartofcompilation instead of truffleboundary in llvmfunctionstartnode
oracle chartype miss type name character
update pmsproductcategorycontroller java
fqn for virtual entities fix
extract components from junit4testbase
stop proximity sensor on pause
declare methods as static
fix exception in byte viewer during table construction
fix potential test painting bug
fix issue with graph mouse controls not working if no listener is set
merge remote tracking branch origin patch
remove user param on userlist method
remove unused method analysis isnostopwords
mute internaltophitstests testreducerandom
add test and some fixes
mute timeserieslifecycleactionsit testforcemergeactionwithcompressioncodec
update support isneedrotatewithsystem with detail
hotfix runtime fix wrong http error code used in exception handling of multipartuploadresource
add missing m in the list of clibraries
fix test failing to recent change
fix datagramchanneltest on windows
merge remote tracking branch origin patch
fileresolvertestbase testresolvefilewithspacesfromclasspath cannot run on windows
lint move declarations into assignments
lint lambda fix
fix always load property when the contextpath is
disable testrequesttimeoutcanceledwhenrequesthasanothererror on windows
use another port to fail connection to
remove race in testdisableidletimeoutinpool
increase timeout in testhttpserverwithidletimeoutsendchunkedfile
move some code
improve test error reporting
fix failing spotbugs test
grid merging caps to avoid loosing the original se options skip ci
merge remote tracking branch origin devel into devel
grid using merged caps to check if video recording was requested skip ci
sulong proper error message if no configuration is applicable
fail methodfilter and immediately when null is passed
simplify xmleventstreamwriter writeendelement
fix checkstyle violation
native image handle speculations on methods that are not part of the image heap
monitor logging fix
regexresult no longer contains a field called groupcount
ruby flavor fix aliasing of curcharclass and charclasstmp
ruby flavor bail out on case folding
ruby flavor parse and bail out on possessive quantifiers
ruby flavor fix bug in unicode property escapes
ruby flavor allow escaping letters with backslash
ruby flavor carriage return is not recognized as a line separator
ruby flavor fix beginning of line assertion semantics
ruby flavor support absent operator expressions
ruby flavor support the ruby syntax for named capture groups
ruby flavor handle escaped parens inside comments
ruby flavor bail out on non ascii byte escapes
ruby flavor fix logic issues in p escape parsing
ruby flavor do not treat after range operator as possessive marker
update pmsskustockcontroller java
fix make copies of the configuration in the environments
ruby flavor bail out on unknown unicode properties
improve javadoc of hostcompilerdirectives
grid fix redirects test in jettyappservertest
java remove unnecessary boxing unboxing for booleans
java deleting unused imports
java fix exception message no functional changes
lpos command add
security disabled for easier experimenting
json utils ext
corrected the changes based on review comment
java temporarily disabling javadoc generation to evaluate how it affects travis build time
redundant logging remove
minor fix formatting
fix load android res map in getter
update umsadmincontroller java
merge remote tracking branch origin devel into devel
add missing license header
hotfix use executorthreadfactory for source coordinator worker threads this deduplicates some code
java fix firefoxoptions copy constructor to fill all the standard capabilities for firefox
add a few suppress warnings
remove unnecessary check in requestmappinghandleradapter
fix case where expansion of groupvertex sends focusedvertex arrow highlight to the origin
remove useless check
hotfix json consider nullability for debezium ingestion timestamp
hotfix table common improve supportswatermarkpushdown javadocs
hotfix core fix javadocs for temporaryclassloadercontext
hotfix runtime minor style warning cleanups in sourcecoordinatorcontext
hotfix runtime replace manual byte reading with datainput readfully
mute watcherutilstests testdeserializesearchrequest
add example to the javadoc
improve file upload test reporting
for code format
java fix ignorance conditions to check if we run in ci environment
for code format
hotfix core extend javadocs for externallyinducedsourcereader
hotfix runtime replace dangling inline javadoc comment by regular comment
ruby flavor fix copyright headers and expand comment on forward references
update umsrolecontroller java
java ignoring a test that fails in github actions environment
remove unused import
java a silly mistake forgot to remove old code that should be replaced by the new one
re enable liveness analysis for all classes
revert change made during debugging
allow netty appserver to respond to head requests
fix loop exit condition for handler registration
fix tests by not declaring the incorrect charset of returned pages
do not crash if recipient_extra is null
integer equals should imply integer not below
fix the windowtest
java ignoring one more test
don t wrap the exceptionininitializererror in an invocationtargetexception
call ensureinitialized in dynamichub newinstance
bugfix copy tab builder
expose mergeloadedsources to runtimes
add new method wasexecuted to optimizedcalltarget
clear voice note media queues within synchronized block
monitor nested states track fix
sql editor refresh actions toolbar on focus gain
disable analysis by default
add singlecontextassumption in llvmpolygloteval
handle more exceptions when querying symbols
make llvmfunctioncode class final
mute xsmoketestpluginsclientyamltestsuiteit while investigating
add option to trigger a young gc before every full gc
csv importer use placeholder name if column name is empty
java add more waits to make distributortest stable
fix rename method in broadcaststream
fix copyright header
minor documentation fix
fix instrumentationhandler visitor bindings bug
grid fix node options test
update cmsprefrenceareaservice java
allow lwjglframe to customize the gdx input instance
grid add warning to show that docker was not available
hotfix runtime expand javadocs and reorder methods in operatorcoordinator
merge branch master of https gitee com linlinjava litemall
java deleting debug code
java actually force nettyserver to use java util logging
java deleting unused private field
monitor logging fix
grid mounting dev shm to browser containers this avoids pages crashing
update cmssubjectservice java
hotfix javadocs fix typo
hotfix fs connector fix the typo in the filesinkitcase
hotfix fs connector remove unused state serializer from filewriter
hotfix connector files add some debug logging to file split enumerator
grid fix allow cors parameter
change group calling feature flag to boolean
use netty s platformdependant util class signed off by julian ladisch eclipse org rtn ladisch de
fix fix genericcli targetoption description msg
check to see if a test should be run before running it
prevent multiple taps from starting multiple calls
enhancer print error exception
do not store reference to class bytes in the symbol table
do verification for patched class constants
update settext to replace newline with space
use proper recipient method in group sync job
update omscompanyaddressservice java
utilize left margin in drawing mask
clear view cache properly after configuration change
fix type in javadoc
ensure we refresh our own profile after refreshing attributes
java fix variable name cyrillic letter с looks exactly like latin letter c
java using try with resources to simplify code
java reducing visibility of a constructor of an abstract class to protected it can be called from a subclass only
java oops to protected not to package private
tracingpolicy is used in public options so it should be annotated too signed off by paulo lopes pmlopes gmail com
fix illegal monitor test
csv importer fix column name
fix string comparison
cleanup nfi usages of exportlibrary
cleanup sl usages of exportlibrary
move invalid interop call from llvmcontext dispose to llvmcontext finalize
do not use newline in vmerror unsupportedfeature error message
add region for locals
fix checkstyle warnings
clear references from the operand stack on exception
remove unused member mappedfieldtype indexanalyzer
update colorutils java add static to colorutils islightcolor
strengthen popframetest to make sure we verify frame arguments when re entering a popped frame simplify stepping for tests
document stale values in substrateinstalledcode and substrateoptimizedcalltarget
if checksums are used print disclaimer
fix typo in javadoc for responseextractor
fix test failing due to timing issue when tree was not yet expanded
fix integer cp patching adds cp patching tests
replace transfertointerpreter with profile
use espressoframe clear to clear locals
unexpected bytecode should never be part of compilation
avoid string conversion on logging
rollback manual hoisting no real improvement measured
disable multi tiered compilation for now
fix word spell
update omsorderreturnreasonservice java
pretty print your phone number in settings
rotate the group calling feature flag
minor optimization as suggested by todo
grid only add docker factories if there is a config for it
use cached native signatures when creating native function wrappers
update omsordersettingservice java
pretty print your phone number in advanced settings
tests merge test timing fix
fix overflow presentation when active speaker changes
update camerax blacklist
prevent surplus notification sound when entering group
fix issues with jumptomessage behavior
remove duplicate property in propertycachekey tostring
fix npe in fastjobstorage getjobcountforqueue
log out exception when a backup fails
fix advanced pin settings pushing wrong fragment
fix another scenario where search position was off
scale sat viewer when it is shown
merge branch master of github com redisson redisson
update parameter description for the result
allow null for result in poly_context_eval poly_value_execute
update ossservice java
fix invocation of method handle getter for static fields
make tier name consistent in tracecompilation output
addressing ecj lints
grid remove unused parameter from dockerflags
java temporarily disabling a piece of code that prevents creation of websockets connections
handle group call update sync messages
make group calling flag hot swappable
llvmsourcelocation getsourcesection should be neverpartofcompilation
hotfix table common deprecate legacy type information conversion methods
hotfix table common fix javadoc typos for functionproviders
hotfix table runtime blink get rowdataserializer from internalserializers
hotfix table runtime blink query converters for internal data structures
auto focus national number field after valid country code in delete fragment
rename method parameter to avoid hiding a field
hotfix javadocs fix typo
fix word spelling
test fix for improper dispose call
fix test failing due to front end status panel change to turn newline characters to spaces
update frame verifier to support new scheme for byte arrays
determine if an integer s square root is an integer in java article by abdallah sawan
remove unnecessary final modifier from static methods
use java style arrays rather than c style arrays
remove redundant extends object
simplify redundant collection operations
replace manual min max calculation with math min
replace loop with arrays fill method call
fix unresolvable symbol in javadoc
replace dangling javadoc comment with block comment
replace lambda with method reference where possible
replace equals expressions with objects equals where possible
remove unnecessary try catch statement
remove redundant close statements
replace atomicreference boolean with atomicboolean in abstractserverhttpresponse writewith
update pmsbrandservice java
update pmsbrandserviceimpl java
fix issue where client deprecation sometimes wasn t cleared
hotfix add nullable annotation to serializedvalue serializeddata
make it an error to reach the default case
remove conversation list datasource throttler
fix compilerasserts neverpartofcompilation reachable in svm
fix missing import
move a common filter to the web package
update newpipe s api url for new app s version check
merge remote tracking branch origin devel into devel
hotfix datadog remove outdated comments
introduce testcontextannotationutils hasannotation
do not add uninstalled extension s lib jars to classpath
merge branch master of github com redisson redisson
update pmsproductattributecategoryservice java
update pmsproductattributeservice java
fix bug recently introduced when updating the table dispose method
avoid using deprecated stringutils isempty method
replace repaint that was carelessly remove
chore delete unused import
hotfix runtime rename schedulertestingutils createscheduler to createschedulerbuilder because it returns a builder
add context to file upload handlers signed off by oscar albrecht oscarfh gmail com
add test to ensure context is propagated in file updates signed off by oscar albrecht oscarfh gmail com
workbench advisor instantiation fix
hotfix coordination add excessresource tostring
document how guest class loading is exposed as a map to interop
add license header
fix set initing false when set freezereason not null signed off by xujie mikawudi qq com
use as separator for method lookup
enable survivor spaces by default
fix npe in client homes panel
fix bad compose input height
force a feature flag refresh after a version change
fix bug with mute states not dynamically updating in participants list
put log tag in brackets
don t rely on the sessionstate protobuf instead use the convenient deserialization constructor
fix not deselecting item that is too large to send
update pmsproductcategoryservice java
update pmsproductcategoryserviceimpl java
replace threadlocal initialvalue with threadlocal withinitial
replace tostring on an array with arrays tostring
remove redundant type argument
remove unused labels
remove getclass call on class instance
remove duplicate param tag
access static method via class reference rather than instance reference
consolidate duplicate branches in switch statement
remove unnecessary return
avoiding npe when the baseuri is not set
grid displaying opentelemetry message only when the tracer is created
used different constructor for backwards compatibility signed off by hakan altindag hakangoudberg hotmail com
hotfix make simpletransformationtranslator methods final they should not be overridden subclasses are meant to implement the internal methods
hotfix fix incorrect output type in streamexectablesourcescan batchexectablesourcescan
hotfix network report channel index if failied to deserialize
hotfix test fix streamconfig propagation to streamtask in streamtaskmailboxtestharnessbuilder
hotfix test improve error message in validatingcheckpointhandler
add method isrefreshing
hotfix remove unused method jobmanagerrunnerimpl isshutdown
hotfix let testingoncompletionactions getjobmasterfailedfuture return jobmasterfailedfuture
hotfix split final and non final fields in jobmaster
fix formatting in reflectiondatabuilder
enable multitier option
rename finalize methods
hotfix coordination remove unused code
fix javadoc link link to the method with javadoc not to the field without
add analysistype findmethod
make amd64arrayindexofnode extensible
remove unused amd64arrayindexofnode constructor
remove unnecessary argument in inlining to deadcodeelimination required is the default
add documentation for outer framestate
make jnigraphkit public
add documentation for instanceofnode with null values
improve jnijavacallwrappermethod documentation
add breakpointnode documentation
fix native image heap writing for hybrid objects
add reflection data getter to afteranalysisaccessimpl
add getters to dynamichub reflection data
make array address lowering public
make indexaddressnode extensible
factor out reflective instantiation java call creation
fix phrasing of documentation
remove unnecessary comment
fix inlineinvokeplugin javadoc wording
fix wording of arrayequalsnode kind documentation
add comment for ifnode optimization of controlflowanchored
update hybrid layout documentation with new identity hashcode location
fix jni getarraylength for hybrid arrays
add support for reusing arraylengthnode of all array allocation nodes
make afterparsecanonicalization phase suite extensible
add toggle for disabling deduplication optimizations on a node
add hook for member inspection before reflection substitution creation
add documentation to jni function tables
update pmsskustockservice java
fix queue display in serviceplayeractivity
fix odd error where using a script keybinding could change a component provider s key binding
small fix to analysis panel to fix size of enablement column
textureatlas textureatlasdata constructor does not accept null use textureatlas to create an empty atlas
add textureatlas load textureatlasdata could load multiple textureatlasdatas into a single textureatlas
initialize random seeds for files at run time
reformat digestauthenticationfeature class
disallow random splittablerandom in the image heap
initialize java lang strictmath randomnumbergeneratorholder at runtime
add missing imports
remove redundant final modifier
add a fast accessor for the java polyglot option
enable isstring substitution and make staticobject s asstring fail if null
fix record clone
grid add implementation flag to the distributor
remove unwanted method
update cmssubjectserviceimpl java
fix checktyle warnings
add a comment for contributors
fix wrong argument in interop writearrayelement
fix staticobject asstring for non string instances
update pmsproductattributecategoryserviceimpl java
remove silenced warning
remove comment the npe is not specific to this method and should be and was detected by unit test
fix npe on dbstypedobject gettypename
merge remote tracking branch origin devel into devel
advanced list set selection on focus
update pmsproductattributeserviceimpl java
fix comment fomatting
add interop s pointer api
add executable interop api
should check for full setting via compareandset also
fix bad exception creation
merge remote tracking branch origin patch
update defaulthttpdatasource java
prevent maxicode sampler from sampling off the edge of the image
add instantiable interop api
allow to instantiate arrays of abstract classes
remove superflous import
