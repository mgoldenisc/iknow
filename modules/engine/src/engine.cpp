#include "engine.h"
#include "UserKnowledgeBase.h"

#include <numeric>
#include <mutex>

#include "Process.h"
#include "SharedMemoryKnowledgebase.h"
#include "CompiledKnowledgebase.h"
#include "IkSummarizer.h"
#include "IkPath.h"
#include "IkIndexInput.h"
#include "IkIndexProcess.h"
#include "RegExServices.h"

const std::set<std::string>& iKnowEngine::GetLanguagesSet(void) {
	static const std::set<std::string> iknow_languages = { "en", "de", "ru", "es", "fr", "ja", "nl", "pt", "sv", "uk", "cs" };
	return iknow_languages;
}

typedef void(*OutputFunc)(iknow::core::IkIndexOutput*, iknow::core::IkIndexDebug<TraceListType>*, void*, Stemmer*);

using namespace std;

using iknow::shell::CProcess;
using iknow::shell::SharedMemoryKnowledgebase;
using iknow::shell::CompiledKnowledgebase;
using iknow::base::String;
using iknow::base::SpaceString;
using iknow::base::Char;
using iknow::base::IkStringEncoding;
using iknow::core::IkSummarizer;
using iknow::core::IkKnowledgebase;
using iknow::core::IkIndexProcess;
using iknow::core::Sentences;
using iknow::core::IkSentence;
using iknow::core::IkMergedLexrep;
using iknow::core::MergedLexreps;
using iknow::core::IkLexrep;
using iknow::core::IkPath;
using iknow::core::FastLabelSet;
using iknow::base::PoolAllocator;
using iknow::core::path::Offsets;
using iknow::core::path::CRC;
using iknow::core::path::CRCs;

using iknowdata::Text_Source;
using iknowdata::Sent_Attribute;
using iknowdata::Entity;
using iknowdata::Path_Attribute;

struct UData
{
	UData(Text_Source::Sentences& sents, iknow::core::IkConceptProximity::ProximityPairVector_t& proximity, std::vector<std::string>& traces) :
		iknow_sentences(sents),
		iknow_proximity(proximity),
		iknow_traces(traces)
	{}

	Text_Source::Sentences& iknow_sentences; // reference to sentence information
	iknow::core::IkConceptProximity::ProximityPairVector_t& iknow_proximity; // collection of proximity information
	std::vector<std::string> &iknow_traces; // linguistic trace info, on demand
};

typedef unsigned short PropertyId;

static iknow::core::RegExServices RegExHandler; // static is fine, API blocks multiple threads...

static const String kEntityString = IkStringEncoding::UTF8ToBase("Entity");
static const String kNegationString = IkStringEncoding::UTF8ToBase("Negation");
static const String kPositiveSentimentString = IkStringEncoding::UTF8ToBase("PositiveSentiment");
static const String kNegativeSentimentString = IkStringEncoding::UTF8ToBase("NegativeSentiment");
static const String kMeasurementString = IkStringEncoding::UTF8ToBase("Measurement"); // Entity(Measurement, Value, Unit), "Value", "Unit", "Value"+"Unit"
static const String kMeasurementValueString = IkStringEncoding::UTF8ToBase("Value");
static const String kMeasurementUnitString = IkStringEncoding::UTF8ToBase("Unit");
static const String kEntityVectorTypeName = IkStringEncoding::UTF8ToBase("EntityVector");

// 
// helper function translates internal iknow::core::IkLabel::Type to iknowdata::Entity::eType
// 
static inline iknowdata::Entity::eType get_ent_type(const IkMergedLexrep *lexrep) {
	switch (lexrep->GetLexrepType()) {
	case iknow::core::IkLabel::Relation: return iknowdata::Entity::Relation;
	case iknow::core::IkLabel::Concept: return iknowdata::Entity::Concept;
	case iknow::core::IkLabel::PathRelevant: return iknowdata::Entity::PathRelevant;
	default: return iknowdata::Entity::NonRelevant;
	}
}

static void iKnowEngineOutputCallback(iknow::core::IkIndexOutput* data, iknow::core::IkIndexDebug<TraceListType>* debug, void* userdata, Stemmer* stemmer)
{
	UData udata = *((UData*)userdata);

	bool bIsIdeographic = data->IsJP();
	const String lexrep_separator = (bIsIdeographic ? String() : SpaceString()); // no separator char for Japanese
	const iknow::base::Char *pText = data->GetTextPointer(); // Get original text pointer

	typedef std::map<const IkMergedLexrep*, std::pair<unsigned short, unsigned short> > mapLexrep2Entity_type;
	mapLexrep2Entity_type mapLexrep2Entity; // map lexreps to entities.

	for (Sentences::iterator i = data->SentencesBegin(); i != data->SentencesEnd(); ++i) {
		IkSentence* sentence = &(*i);
		const IkKnowledgebase* kb = sentence->GetLexrepsBegin()->LexrepsBegin()->GetKnowledgebase(); // KB does not change in a sentence.
		RegExHandler.swich_kb(kb); // Switch to the correct KB

		iknowdata::Sentence sentence_data;
		for (MergedLexreps::const_iterator j = sentence->GetLexrepsBegin(); j != sentence->GetLexrepsEnd(); ++j) { // iterate entities
			const IkMergedLexrep *lexrep = &(*j);

			// map lexrep to sentence and entity
			unsigned short idx_sentence = static_cast<unsigned short>(udata.iknow_sentences.size()); // sentence reference
			unsigned short idx_entity = static_cast<unsigned short>(sentence_data.entities.size()); // entity reference
			mapLexrep2Entity.insert(make_pair(lexrep, make_pair(idx_sentence, idx_entity))); // link ID to lexrep

			const Char* literal_start = lexrep->GetTextPointerBegin();
			const Char* literal_end = lexrep->GetTextPointerEnd();
			const size_t text_start = literal_start - pText;
			const size_t text_stop = literal_end - pText;

			{
				// handle measurement attributes
				int iPushMeasurementAttribute = 0;
				for (IkMergedLexrep::const_iterator it = lexrep->LexrepsBegin(); it != lexrep->LexrepsEnd(); it++) { // Scan for label attributes : scroll the single lexreps
					std::string a_marker = iknow::base::IkStringEncoding::BaseToUTF8(it->GetValue()); // the attribute marker, Literal representation

					bool bMarkerIsMeasure = false, bMarkerIsValue = false, bMarkerIsUnit = false;
					const size_t label_count = it->NumberOfLabels();
					for (size_t i = 0; i < label_count; ++i) {
						FastLabelSet::Index idx_label = it->GetLabelIndexAt(i);
						size_t attribute_count = kb->GetAttributeCount(idx_label); // how many attributes on this label ?
						for (size_t cnt_attribute = 0; cnt_attribute < attribute_count; cnt_attribute++) { // iterate the attributes
							iknow::core::AttributeId type_attribute = kb->GetAttributeType(idx_label, cnt_attribute);
							String name_attribute(kb->AttributeNameForId(type_attribute).data, kb->AttributeNameForId(type_attribute).size);

							if (name_attribute == kEntityString) { // found an entity attribute, read the parameters
								for (const iknow::core::AttributeId* ent_param = kb->GetAttributeParamsBegin(idx_label, cnt_attribute); ent_param != kb->GetAttributeParamsEnd(idx_label, cnt_attribute); ++ent_param) {
									String param_attribute(kb->AttributeNameForId(*ent_param).data, kb->AttributeNameForId(*ent_param).size);
									iknow::core::PropertyId id_property = kb->PropertyIdForName(param_attribute);
									iknowdata::Attribute a_type = static_cast<iknowdata::Attribute>(id_property);
									if (a_type == iknowdata::Attribute::Measurement) // measurement attribute
										bMarkerIsMeasure = true;
									if (bMarkerIsMeasure) {
										if (param_attribute == kMeasurementValueString) { // measurement value attribute type
											bMarkerIsValue = true;
										}
										if (param_attribute == kMeasurementUnitString) { // measurement unit attribute type
											bMarkerIsUnit = true;
										}
									}
								}
							}
						}
					}
					if (bMarkerIsMeasure) { // collect for merged lexrep
						if (!iPushMeasurementAttribute) {
							sentence_data.sent_attributes.push_back(Sent_Attribute(iknowdata::Attribute::Measurement, it->GetTextPointerBegin() - pText, it->GetTextPointerEnd() - pText, a_marker)); // write marker info
							sentence_data.sent_attributes.back().entity_ref = static_cast<unsigned short>(sentence_data.entities.size()); // connect sentence attribute to entity
						}
						if (iPushMeasurementAttribute > 0) { // adjust marker, except for the first.
							sentence_data.sent_attributes.back().marker_ += " " + a_marker;
							sentence_data.sent_attributes.back().offset_stop_ = it->GetTextPointerEnd() - pText;
						}
						// handle value & unit
						String at_value, at_unit, at_value2, at_unit2; // measurement attribute properties
						String at_token = it->GetValue();
						if (bMarkerIsValue && bMarkerIsUnit) { // separate value & unit properties
							if (!RegExHandler.SplitValueUnit(it->GetNormalizedText(), at_value, at_unit)) {  // failed to separate
								int parts = RegExHandler.Parser2(it->GetNormalizedText(), at_value, at_unit, at_value2, at_unit2); // refined value/unit parser
								if (parts == 0) at_value = it->GetNormalizedText(); // set as Value
							}
						}
						else {
							if (bMarkerIsValue) at_value = it->GetNormalizedText();
							if (bMarkerIsUnit) at_unit = it->GetNormalizedText();
						}
						Sent_Attribute& ref = sentence_data.sent_attributes.back();
						if (at_value.length()) {
								if (ref.value_.empty()) // fill up first placeholder
									ref.value_ = iknow::base::IkStringEncoding::BaseToUTF8(at_value);
								else {
									if (ref.value2_.empty()) // fill up second placeholder
										ref.value2_ = iknow::base::IkStringEncoding::BaseToUTF8(at_value);
								}
							}
							if (at_unit.length()) {
								if (ref.unit_.empty() && ref.value2_.empty()) // if we have a second value, write the second unit
									ref.unit_ = iknow::base::IkStringEncoding::BaseToUTF8(at_unit);
								else {
									if (ref.unit2_.empty()) // fill up second  placeholder
										ref.unit2_ = iknow::base::IkStringEncoding::BaseToUTF8(at_unit);
								}
						}
						if (at_value2.length()) {
							if (ref.value2_.empty()) // only write if available
								ref.value2_ = iknow::base::IkStringEncoding::BaseToUTF8(at_value2);
						}
						if (at_unit2.length()) {
							if (ref.unit2_.empty()) // only write if available
								ref.unit2_ = iknow::base::IkStringEncoding::BaseToUTF8(at_unit2);
						}
						iPushMeasurementAttribute++;
					}
				}
			}
			std::map<iknowdata::Attribute, Sent_Attribute*> mapLexrepAttributes;
			for (IkMergedLexrep::const_iterator it = lexrep->LexrepsBegin(); it != lexrep->LexrepsEnd(); it++) { // Scan for label attributes : scroll the single lexreps
				std::string a_marker = iknow::base::IkStringEncoding::BaseToUTF8(it->GetValue()); // the attribute marker, Literal representation
				char certainty_data = it->GetCertainty(); // certainty char, "0" to "9"

				const size_t label_count = it->NumberOfLabels();
				for (size_t i = 0; i < label_count; ++i) {
					FastLabelSet::Index idx_label = it->GetLabelIndexAt(i);
					size_t attribute_count = kb->GetAttributeCount(idx_label); // how many attributes on this label ?
					for (size_t cnt_attribute = 0; cnt_attribute < attribute_count; cnt_attribute++) { // iterate the attributes
						iknow::core::AttributeId type_attribute = kb->GetAttributeType(idx_label, cnt_attribute);
						String name_attribute(kb->AttributeNameForId(type_attribute).data, kb->AttributeNameForId(type_attribute).size);

						if (name_attribute == kEntityString) { // found an entity attribute, read the parameters							
							iknow::core::PropertyId id_property = 0; // first parameter is property
							for (const iknow::core::AttributeId* ent_param = kb->GetAttributeParamsBegin(idx_label, cnt_attribute); ent_param != kb->GetAttributeParamsEnd(idx_label, cnt_attribute); ++ent_param) {
								String param_attribute(kb->AttributeNameForId(*ent_param).data, kb->AttributeNameForId(*ent_param).size);
								if (!id_property) 
									id_property = kb->PropertyIdForName(param_attribute);
							}
							iknowdata::Attribute a_type = static_cast<iknowdata::Attribute>(id_property);
							if (a_type != iknowdata::Attribute::Measurement) { // measurements are handled separately
								if (mapLexrepAttributes.count(a_type)) { // continue existing
									Sent_Attribute* p_att = mapLexrepAttributes[a_type];

									p_att->offset_stop_ = it->GetTextPointerEnd() - pText; // enlarge the literal
									p_att->marker_ += string(" ") + a_marker; // add the extra marker
								}
								else { // new
									sentence_data.sent_attributes.push_back(Sent_Attribute(a_type, it->GetTextPointerBegin() - pText, it->GetTextPointerEnd() - pText, a_marker)); // write marker info
									sentence_data.sent_attributes.back().entity_ref = static_cast<unsigned short>(sentence_data.entities.size()); // connect sentence attribute to entity
									if (id_property == IKATTCERTAINTY) { // certainty attribute, emit the certainty level
										if (certainty_data != '\0') {
											std::string certainty_value = "0";
											certainty_value[0] = certainty_data; // '0' to '9'
											Sent_Attribute& ref = sentence_data.sent_attributes.back();
											ref.value_ = certainty_value;
										}
									}
									mapLexrepAttributes.insert(make_pair(a_type, &sentence_data.sent_attributes.back())); // link lexrep attribute to data structure
								}
							}
						}
					}
				}				
			}
			std::string index_value = IkStringEncoding::BaseToUTF8(lexrep->GetNormalizedText());
			iknow::core::IkIndexOutput::EntityId ent_id = data->GetEntityID(lexrep);
			iknowdata::Entity::eType e_type = get_ent_type(lexrep);
			sentence_data.entities.push_back(Entity(e_type, text_start, text_stop, index_value, data->GetEntityDominance(lexrep), ent_id));
		}
		// collect sentence path data
		if (bIsIdeographic) { // handle entity vectors
			// iknow::core::PropertyId entity_vector_prop_id = kb->PropertyIdForName(kEntityVectorTypeName); // entity vector property
			IkSentence::EntityVector& entity_vector = sentence->GetEntityVector();
			if (!entity_vector.empty()) { // emit entity vectors
				// Sent_Attribute::aType a_type = static_cast<Sent_Attribute::aType>(entity_vector_prop_id);

				for (IkSentence::EntityVector::const_iterator i = entity_vector.begin(); i != entity_vector.end(); ++i) { // collect entity id's
					// (*this)(*i + 1); //occurrence ids are 1-based
					sentence_data.path.push_back((unsigned short)*i); 
				}
			}
		}
		else { // normal path
			{	// collect path attribute expansions
				DirectOutputPaths& sent_paths = data->paths_[udata.iknow_sentences.size()]; // paths for the sentence (in fact, only one per sentence after introducing path_relevants)
				for (DirectOutputPaths::iterator it_path = sent_paths.begin(); it_path != sent_paths.end(); ++it_path) // iterate all paths (in fact, only one...)
				{
					size_t path_length = it_path->offsets.size();
					for (int i = 0; i < path_length; ++i) {
						const IkMergedLexrep* lexrep = it_path->offsets[i];
						unsigned short entity_id = mapLexrep2Entity[lexrep].second; // entity id from lexrep
						sentence_data.path.push_back(entity_id); // reference to sentence entities
					}
					DirectOutputPathAttributeMap& amap = it_path->attributes;
					for (DirectOutputPathAttributeMap::iterator it_attr = amap.begin(); it_attr != amap.end(); ++it_attr) { // iterate per attribute id
						PropertyId id_attr = it_attr->first;
						DirectOutputPathAttributes& path_attr = it_attr->second;
						for (DirectOutputPathAttributes::iterator it_path_attr = path_attr.begin(); it_path_attr != path_attr.end(); ++it_path_attr) { // iterate per attribute id path
							DirectOutputPathAttribute& pa = *it_path_attr; // single attribute id path = path attribute expansion
							PropertyId id_attr_path = pa.type; // is equal to "id_attr"
							PathOffset attr_path_begin = pa.begin; // refers to path
							PathOffset attr_path_end = pa.end; // refers to path
							// cout << id_attr_path << ":" << attr_path_begin << ":" << attr_path_end << std::endl;
							Path_Attribute path_attribute;
							path_attribute.type = static_cast<iknowdata::Attribute>(id_attr_path);
							path_attribute.pos = (unsigned short) pa.begin; // start position
							path_attribute.span = (unsigned short) (pa.begin == pa.end ? 1 : pa.end - pa.begin); // attribute expansion span, minimum = 1
							sentence_data.path_attributes.push_back(path_attribute);
						}
					}
				}
			}
		}
		udata.iknow_sentences.push_back(sentence_data); // Collect single sentence data
	}
	data->GetProximityPairVector(udata.iknow_proximity); // Proximity is document related

	if (debug) {
		const iknow::base::IkTrace<Utf8List>& trace_data = debug->GetTrace();
		udata.iknow_traces.reserve(trace_data.end() - trace_data.begin()); // reserve memory for storage vector
		for (iknow::base::IkTrace<Utf8List>::Items::const_iterator it = trace_data.begin(); it != trace_data.end(); ++it) {
			const String& key = it->first;
			const Utf8List& valueList = it->second;
			string value;
			for_each(valueList.begin(), valueList.end(), [&value](const string& item) { value += (item + ";"); });

			udata.iknow_traces.push_back(IkStringEncoding::BaseToUTF8(key) + ":" + value);
		}
	}
}

static SharedMemoryKnowledgebase *pUserDCT = NULL; // User dictionary, one per process

iKnowEngine::iKnowEngine() // Constructor
{
}

iKnowEngine::~iKnowEngine() // Destructor
{
}

typedef std::map<iknow::base::String, iknow::core::IkKnowledgebase*> KnowledgebaseMap;

extern "C" {
	extern const unsigned char kb_en_data[];
	extern const unsigned char kb_de_data[];
	extern const unsigned char kb_ru_data[];
	extern const unsigned char kb_es_data[];
	extern const unsigned char kb_fr_data[];
	extern const unsigned char kb_ja_data[];
	extern const unsigned char kb_nl_data[];
	extern const unsigned char kb_pt_data[];
	extern const unsigned char kb_sv_data[];
	extern const unsigned char kb_uk_data[];
	extern const unsigned char kb_cs_data[];
}

struct LanguageCodeMap {
	LanguageCodeMap() {
		map.insert(CodeMap::value_type("en", const_cast<unsigned char *>(&(kb_en_data[0]))));
		map.insert(CodeMap::value_type("de", const_cast<unsigned char *>(&(kb_de_data[0]))));
		map.insert(CodeMap::value_type("ru", const_cast<unsigned char *>(&(kb_ru_data[0]))));
		map.insert(CodeMap::value_type("es", const_cast<unsigned char *>(&(kb_es_data[0]))));
		map.insert(CodeMap::value_type("fr", const_cast<unsigned char *>(&(kb_fr_data[0]))));
		map.insert(CodeMap::value_type("ja", const_cast<unsigned char *>(&(kb_ja_data[0]))));
		map.insert(CodeMap::value_type("nl", const_cast<unsigned char *>(&(kb_nl_data[0]))));
		map.insert(CodeMap::value_type("pt", const_cast<unsigned char *>(&(kb_pt_data[0]))));
		map.insert(CodeMap::value_type("sv", const_cast<unsigned char *>(&(kb_sv_data[0]))));
		map.insert(CodeMap::value_type("uk", const_cast<unsigned char *>(&(kb_uk_data[0]))));
		map.insert(CodeMap::value_type("cs", const_cast<unsigned char *>(&(kb_cs_data[0]))));
	}
	unsigned char *Lookup(const std::string& language_name) const {
		CodeMap::const_iterator i = map.find(language_name);
		return ((i == map.end()) ? NULL : i->second);
	}

	typedef std::map<std::string, unsigned char *> CodeMap;
	CodeMap map;
};

const static LanguageCodeMap language_code_map;
static std::mutex mtx;           // mutex for process.IndexFunc critical section


void iKnowEngine::index(iknow::base::String& text_input, const std::string& utf8language, bool b_trace)
{
	if (GetLanguagesSet().count(utf8language) == 0) // language not supported
		throw ExceptionFrom<iKnowEngine>("Language not supported");

	std::unique_lock<std::mutex> lck(mtx, std::defer_lock);

	m_index.sentences.clear();
	m_index.proximity.clear();
	m_traces.clear();
	UData udata(m_index.sentences, m_index.proximity, m_traces);

	SharedMemoryKnowledgebase skb = language_code_map.Lookup(utf8language);

	CompiledKnowledgebase ckb(&skb, utf8language);
	CProcess::type_languageKbMap temp_map;
	temp_map.insert(CProcess::type_languageKbMap::value_type(IkStringEncoding::UTF8ToBase(utf8language), &ckb));
	CProcess process(temp_map);

	lck.lock(); // critical section (exclusive access to IndexFunc by locking lck):

	process.setUserDictionary(pUserDCT);
	if (pUserDCT)
		pUserDCT->FilterInput(text_input); // rewritings can only be applied if we no longer emit text offsets

	iknow::core::IkIndexInput Input(&text_input);
	process.IndexFunc(Input, iKnowEngineOutputCallback, &udata, true, b_trace);
	lck.unlock();
}

void iKnowEngine::index(const std::string& text_source, const std::string& language, bool b_trace) {
	String text_source_ucs2(IkStringEncoding::UTF8ToBase(text_source));
	index(text_source_ucs2, language, b_trace);
}

std::string iKnowEngine::NormalizeText(const string& text_source, const std::string& language, bool bUserDct, bool bLowerCase, bool bStripPunct) {
	try {
		SharedMemoryKnowledgebase skb = language_code_map.Lookup(language); //No ALI for normalization: We need a KB.
		IkKnowledgebase* kb = &skb;
		IkKnowledgebase* ud_kb = NULL;

		std::map<iknow::base::String, IkKnowledgebase const*> null_kb_map;
		IkIndexProcess process(null_kb_map);
		String output = process.NormalizeText(IkStringEncoding::UTF8ToBase(text_source), kb, ud_kb, bLowerCase, bStripPunct);
		return IkStringEncoding::BaseToUTF8(output);
	}
	catch (const std::exception& e) {
		throw ExceptionFrom<iKnowEngine>(e.what());
	}
	throw std::runtime_error("Failed to throw an exception.");
}

// Constructor
UserDictionary::UserDictionary() {
}

// Clear the User Dictionary object.
void UserDictionary::clear() {
	m_user_data.clear();
}

// Adds User Dictionary label to a lexical representation for customizing purposes
int UserDictionary::addLabel(const std::string& literal, const char* UdctLabel)
{
	string normalized = iKnowEngine::NormalizeText(literal, "en"); // normalize the literal
	if (m_user_data.addLexrepLabel(normalized, UdctLabel) == -1) return iKnowEngine::iknow_unknown_label; // add to the udct lexreps
	return 0; // OK
}

// Add User Dictionary literal rewrite, not functional.
void UserDictionary::addEntry(const std::string& literal, const string& literal_rewrite) {
	return; // we cannot rewrite input text as long as we use text offsets to annotate text.
}

// Add User Dictionary EndNoEnd. 
void UserDictionary::addSEndCondition(const std::string& literal, bool b_end)
{
	m_user_data.addSEndCondition(literal, b_end);
}

// Shortcut for known UD labels
void UserDictionary::addConceptTerm(const std::string& literal) {
	addLabel(literal, "UDConcept");
}
void UserDictionary::addRelationTerm(const std::string& literal) {
	addLabel(literal, "UDRelation");
}
void UserDictionary::addNonrelevantTerm(const std::string& literal) {
	addLabel(literal, "UDNonRelevant");
}
void UserDictionary::addUnitTerm(const std::string& literal) {
	addLabel(literal, "UDUnit");
}
void UserDictionary::addNumberTerm(const std::string& literal) {
	addLabel(literal, "UDNumber");
}
void UserDictionary::addTimeTerm(const std::string& literal) {
	addLabel(literal, "UDTime");
}
void UserDictionary::addNegationTerm(const std::string& literal) {
	addLabel(literal, "UDNegation");
}
void UserDictionary::addPositiveSentimentTerm(const std::string& literal) {
	addLabel(literal, "UDPosSentiment");
}
void UserDictionary::addNegativeSentimentTerm(const std::string& literal) {
	addLabel(literal, "UDNegSentiment");
}

// Load a User Dictionary into the iKnow engine
void iKnowEngine::loadUserDictionary(UserDictionary& udct)
{
	unloadUserDictionary(); // we can only have one
	pUserDCT = new SharedMemoryKnowledgebase(udct.m_user_data.generateRAW(false));	// first use generation.
}

// Unload the User Dictionary
void iKnowEngine::unloadUserDictionary(void)
{
	if (pUserDCT != NULL) {
		delete pUserDCT;
		pUserDCT = NULL;
	}
}